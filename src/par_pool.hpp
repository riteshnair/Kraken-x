// ============================================================================
//  par_pool.hpp — persistent worker pool for the CPU backend.
//
//  The reference backend was single-threaded by design; on a many-core
//  host that leaves the machine idle. This pool parallelizes the
//  BLAS-shaped ops across INDEPENDENT units of work (output columns,
//  attention heads, token rows). No reduction is ever split across
//  blocks, so every floating-point accumulation keeps its serial order
//  and results are bit-identical to the single-threaded path — the
//  backend stays a valid oracle for the GPU implementation.
//
//  Sizing: one worker per hardware thread minus the calling thread, or
//  KRK_THREADS in the environment. EngineConfig::threads can pin it via
//  set_workers() before the first forward pass.
//
//  Scheduling. Every block in one par_for does identical work (same
//  grain, same op), so ranges are partitioned STATICALLY: participant
//  p owns blocks [p*stride, (p+1)*stride). There is no shared block
//  counter on the hot path — a decode step issues ~200 par_for calls
//  back to back, and making 32 threads fight over one mutex (or worse,
//  a condition variable) per op costs more than the work itself; that
//  contention is why decode degraded at high thread counts before.
//  Each participant reports completion with a single lock per task.
//
//  Wake policy — spin, then park. Ops arrive in bursts, so workers
//  watch the atomic task id for a couple of milliseconds before
//  parking: back-to-back ops find a hot team with zero syscalls,
//  while long idle periods still sleep.
// ============================================================================
#ifndef KRK_PAR_POOL_HPP
#define KRK_PAR_POOL_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "krk/common.hpp"

namespace krk {

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
inline void cpu_relax() { __builtin_ia32_pause(); }
#elif defined(_MSC_VER)
inline void cpu_relax() { _mm_pause(); }
#else
inline void cpu_relax() {}
#endif
#else
inline void cpu_relax() {}
#endif

// ============================================================================
//  Core placement.
//
//  Every thread the pool spawns is an OS thread, and the OS is free to move it
//  at any time: through a whole GEMM a worker can land on a slow core, share a
//  core with another worker, or follow a migrating interrupt. On a hybrid part
//  (Zen 4/5, Alder/Raptor Lake, Snapdragon X) it is worse than that — the fast
//  cores are not enumerated first, so `hardware_concurrency()` threads spread
//  over the slow cores and the *whole* pool runs at the E-core clock. That is
//  the single cheapest throughput a CPU backend can leave on the table, and it
//  costs one syscall per thread at startup.
//
//  Policy, fastest core first:
//    * the calling thread (which also runs a slice) gets the best core;
//    * worker i gets the (i+1)-th best core, so a small pool lands entirely on
//      the performance cluster instead of straddling both;
//    * a pool larger than the machine wraps — sharing is better than parking.
//
//  The ordering comes from cpufreq, which is the only portable way to tell the
//  clusters apart without a vendor SDK (Apple exposes nothing at all; Windows
//  would need GetSystemCpuSetInformation). No cpufreq — a VM, a container, an
//  old kernel, a CPU without the driver — means no order to impose, and the
//  pool keeps its previous unpinned behaviour. KRK_AFFINITY=1 forces pinning on
//  such a box anyway (useful to measure placement alone), KRK_AFFINITY=0 always
//  disables it.
// ============================================================================
struct CpuTopology {
    std::vector<i64> order;  // CPU ids, fastest first
    i64 n_fast = 0;          // cores within 15% of the fastest clock
    i64 n_slow = 0;          // the rest, when the part is hybrid
    i64 max_khz = 0;
    bool hybrid = false;
    bool pinned = false;
    // LOGICAL is not PHYSICAL: n_fast counts SMT siblings as threads (8 on a
    // 4P+8E+2LP-E Meteor Lake), while n_fast_phys counts the physical P cores
    // (4). Which one the POOL should use is a measurement, and it is NOT the
    // physical count -- see the width table in the Pool constructor. This field
    // is kept because a caller that wants "one participant per physical core"
    // has no other way to ask; the pool itself does not cap to it.
    i64 n_fast_phys = 0;

    std::string summary() const {
        if (order.empty()) return "cpu topology unknown";
        std::string s = format("%lld cores", static_cast<long long>(order.size()));
        if (max_khz > 0) {
            s += format(", top %.2f GHz", max_khz / 1e6);
            if (hybrid)
                s += format(", %lldP+%lldE", static_cast<long long>(n_fast),
                                static_cast<long long>(n_slow));
            else
                s += " (uniform)";
        }
        s += pinned ? ", pinned" : ", unpinned";
        return s;
    }
};

inline i64 cpu_max_khz(i64 cpu) {
#if defined(__linux__)
    static const char *const sinks[] = {
        "/sys/devices/system/cpu/cpu%lld/cpufreq/cpuinfo_max_freq",
        "/sys/devices/system/cpu/cpu%lld/cpufreq/scaling_max_freq",
    };
    for (const char *pat : sinks) {
        char path[192];
        std::snprintf(path, sizeof path, pat, static_cast<long long>(cpu));
        if (FILE *f = std::fopen(path, "r")) {
            long long v = 0;
            if (std::fscanf(f, "%lld", &v) == 1) {
                std::fclose(f);
                if (v > 0) return v;
                continue;
            }
            std::fclose(f);
        }
    }
#else
    (void)cpu;
#endif
    return 0;
}

#if defined(_WIN32)
// Windows hybrid detection: EfficiencyClass from RelationProcessorCore is the
// only portable signal (higher = P/performance on Intel 12th+ and Qualcomm X).
// Returns true and fills t when it can order logical processors; false leaves
// the caller on the identity fallback. Single-group only (< 64 logical CPUs),
// which covers every client hybrid this engine targets.
inline bool detect_cpu_topology_win(CpuTopology &t, i64 hc) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return false;
    std::vector<unsigned char> buf(len);
    auto *info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return false;

    struct Lp { int id; BYTE eff; };
    std::vector<Lp> lps;
    std::vector<BYTE> core_eff;  // one entry per PHYSICAL core
    lps.reserve(static_cast<size_t>(hc));
    for (unsigned char *p = buf.data(); p < buf.data() + len; ) {
        auto *e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(p);
        if (e->Relationship == RelationProcessorCore) {
            const BYTE eff = e->Processor.EfficiencyClass;
            core_eff.push_back(eff);
            for (WORD g = 0; g < e->Processor.GroupCount; g++) {
                const auto &gm = e->Processor.GroupMask[g];
                KAFFINITY mask = gm.Mask;
                for (int b = 0; b < 64 && mask; b++, mask >>= 1) {
                    if (mask & 1) lps.push_back({static_cast<int>(gm.Group) * 64 + b, eff});
                }
            }
        }
        p += e->Size;
    }
    if (lps.empty()) return false;
    std::stable_sort(lps.begin(), lps.end(), [](const Lp &a, const Lp &b) {
        if (a.eff != b.eff) return a.eff > b.eff; // P cores first
        return a.id < b.id;
    });
    BYTE max_eff = 0, min_eff = 255;
    for (const auto &lp : lps) { if (lp.eff > max_eff) max_eff = lp.eff; if (lp.eff < min_eff) min_eff = lp.eff; }
    t.order.clear();
    t.order.reserve(lps.size());
    t.n_fast = 0; t.n_slow = 0; t.max_khz = 0;
    // Split by efficiency class, not by clock (unavailable here): the top class
    // is the performance cluster. A two-class split (P+E) is the common case;
    // three classes (P+E+LP-E, Meteor Lake) fold LP-E into "slow".
    for (const auto &lp : lps) {
        t.order.push_back(static_cast<i64>(lp.id));
        if (lp.eff == max_eff) t.n_fast++; else t.n_slow++;
    }
    t.hybrid = max_eff != min_eff;
    // One RelationProcessorCore entry per PHYSICAL core, so the count of
    // top-class entries is the physical P-core count, not the LP count.
    t.n_fast_phys = 0;
    for (BYTE e : core_eff)
        if (e == max_eff) t.n_fast_phys++;
    // If the OS reports more logical CPUs than we enumerated (multi-group box
    // we don't fully map), keep the enumeration as the preferred prefix and do
    // not claim a complete topology.
    if (static_cast<i64>(t.order.size()) < hc && !t.hybrid) return false;
    return true;
}
#endif

inline CpuTopology detect_cpu_topology(i64 hc) {
    CpuTopology t;
    t.order.reserve(static_cast<size_t>(hc));
#if defined(_WIN32)
    if (detect_cpu_topology_win(t, hc)) return t;
    t.order.clear();
#endif
    std::vector<std::pair<i64, i64>> f;  // (khz, cpu)
    f.reserve(static_cast<size_t>(hc));
    for (i64 i = 0; i < hc; i++) {
        const i64 k = cpu_max_khz(i);
        if (k > 0) {
            f.emplace_back(k, i);
            if (k > t.max_khz) t.max_khz = k;
        }
    }
    if (static_cast<i64>(f.size()) == hc && hc > 1) {
        // Stable: equal clocks keep their CPU order, so a uniform part gets the
        // identity permutation and nothing moves.
        std::stable_sort(f.begin(), f.end(),
                         [](const std::pair<i64, i64> &a, const std::pair<i64, i64> &b) {
                             return a.first > b.first;
                         });
        for (const auto &kv : f) {
            t.order.push_back(kv.second);
            // 15%: absorbs DVFS rounding and SMT siblings, still separates a
            // 3.0 GHz P core from a 2.0 GHz E core.
            if (kv.first * 100 >= t.max_khz * 85) t.n_fast++;
            else t.n_slow++;
        }
        t.hybrid = t.n_slow > 0;
        // No per-core view on this path, so the logical count is the only
        // answer available; treating it as physical would cap every small op at
        // the P+HT width, which is what the engine already did. Leave 0 (no
        // cap) so an un-inspectable box keeps its previous behaviour.
        t.n_fast_phys = 0;
    } else {
        for (i64 i = 0; i < hc; i++) t.order.push_back(i);
        t.n_fast = hc;
        t.n_fast_phys = 0;
    }
    return t;
}

inline bool pin_this_thread(i64 cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#elif defined(_WIN32)
    if (cpu < 64) {
        SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
        return true;
    }
    return false;
#else
    (void)cpu;
    return false;
#endif
}

class Pool {
public:
    static Pool &get() {
        static Pool pool;
        return pool;
    }

    // Total participants (workers + the calling thread).
    i64 workers() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<i64>(workers_.size()) + 1;
    }

    // Cap the number of participants used by par_for (workers + calling thread).
    // 0 (default) means all available. Values > available are clamped at run
    // time. This does not retire workers: it is cheap to change between phases
    // (e.g. prefill vs decode) and must only be called when no task is in
    // flight. With static partitioning, workers are assigned slot indices
    // 0..participants-2 and the caller takes slot participants-1, so a cap
    // uses the fastest-prefix participants (P cores first on hybrid).
    void set_active_cap(i64 n) {
        active_cap_.store(n > 0 ? n : 0, std::memory_order_release);
    }
    i64 active_cap() const {
        return active_cap_.load(std::memory_order_acquire);
    }

    // Which core each participant landed on. Settled at construction and
    // re-settled by set_workers(), so it is safe to read any time.
    const CpuTopology &topology() const { return topo_; }
    // Physical performance-core count, when the OS reported one. This is the
    // width at which a decode-shaped op stops waiting on an SMT sibling or an
    // E-core. 0 means unknown -> callers apply no cap.
    i64 physical_fast() const { return topo_.n_fast_phys; }

    // Pin the total participant count (workers = n - 1). Retires the
    // current worker generation and spawns a new one; a no-op while a
    // task is in flight. The engine calls this during init, before the
    // first forward pass.
    void set_workers(i64 n) {
        // THE JOIN MUST NOT HOLD mu_. A parked worker is inside
        // cv_start_.wait(lk, pred), and re-checking that predicate means
        // reacquiring mu_ -- so joining it while this thread owns mu_ waits for
        // a thread that is waiting for the mutex. That is a hard deadlock, and
        // it only fired sometimes because a worker that was still in its 2 ms
        // SPIN phase observes the generation bump without touching mu_ at all:
        // measured with `--threads 4` on Qwen3.8-4B, 4 hangs in 12 runs, while
        // the same width set through KRK_THREADS (which sizes the pool in the
        // constructor and never calls this) hung 0 times in 12.
        //
        // So: bump the generation and take the old generation out of
        // workers_ under the lock, notify, release, and only then join.
        std::vector<std::thread> old;
        i64 want = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (n < 1 || body_fn_ != nullptr) return;
            want = std::min<i64>(n - 1, 63);
            if (want == static_cast<i64>(workers_.size())) return;
            gen_.fetch_add(1, std::memory_order_release);
            old.swap(workers_);
            cv_start_.notify_all();
        }
        for (std::thread &t : old)
            if (t.joinable()) t.join();
        old.clear();
        // No task can be in flight here (body_fn_ == nullptr on entry, and a
        // worker's only way back in is a task id this thread has not
        // published), so the new generation starts at the CURRENT id: seeding
        // it with 0 would send every fresh worker chasing a task that was
        // drained long ago.
        const i64 start = task_id_.load(std::memory_order_acquire);
        std::lock_guard<std::mutex> lk(mu_);
        for (i64 i = 0; i < want; i++)
            workers_.emplace_back(
                [this, g = gen_.load(std::memory_order_acquire), idx = i, start] {
                    // The constructor pins each worker to the core it was
                    // assigned; this path did not, so `--threads N` silently
                    // gave up the fastest-first placement while the banner went
                    // on reporting "pinned". Same call, same width.
                    if (pinned_) pin_this_thread(worker_cpu(idx));
                    worker_loop(g, idx, start);
                });
    }

    // Run body(b, e) over blocks of at most `grain` indices covering
    // [0, n). body must touch only data disjoint from other blocks and
    // must not call back into the pool.
    //
    // `split == false` forces the serial path: the flop floor in par_for is a
    // throughput policy, not a semantic one. Both branches below call the SAME
    // instantiation of `invoke`, and that is load-bearing rather than tidy.
    //
    // Calling the lambda directly at the call site (what this did for small
    // ops) and calling it through std::function inside the pool (what it did
    // for large ones) compiles the same arithmetic TWICE, and GCC contracts
    // `a * c - b * s` into an FMA in one copy and not in the other. The
    // rounding difference is ~1e-7 relative, which is enough to make the CPU
    // reference backend's answer depend on how many rows the caller passed in
    // one call: measured on RoPE, row 65 of a 66-row call differed from the
    // same row of a 1-row call in 200 of 512 lanes, and that is exactly the
    // chunk-invariance property test_prefill_chunk_bitexact gates on. One
    // trampoline for both paths makes the serial case and the pooled case the
    // same machine code, so the floor can be tuned without moving a number.
    template <class F>
    void run(i64 n, i64 grain, bool split, F &&body, i64 max_parts = 0) {
        if (n <= 0) return;
        using G = typename std::remove_reference<F>::type;
        void (*const invoke)(void *, i64, i64) = [](void *ctx, i64 b, i64 e) {
            (*static_cast<G *>(ctx))(b, e);
        };
        if (grain < 1) grain = n < 1 ? 1 : n;
        const i64 blocks = (n + grain - 1) / grain;
        const i64 all = static_cast<i64>(workers_.size()) + 1;
        i64 participants = all;
        const i64 cap = active_cap_.load(std::memory_order_acquire);
        if (cap > 0 && cap < participants) participants = cap;
        // Per-call ceiling (see par_for's kParSmallFloor): a small op that is
        // split wider than the physical performance cores spends its time
        // waiting for participants that are slower than the ones it already
        // has. `max_parts == 0` means no ceiling.
        if (max_parts > 0 && max_parts < participants) participants = max_parts;
        if (!split || blocks <= 1 || participants <= 1) {
            invoke(&body, 0, n);
            return;
        }
        // Static partition of the (uniform) blocks. stride >= 1 even
        // when blocks < participants; participants whose range is empty
        // simply do nothing. Worker slots 0..participants-2 and the
        // calling thread as slot participants-1 take part; any workers
        // with index >= participants-1 are idle for this task.
        const i64 stride = (blocks + participants - 1) / participants;
        // The calling thread is the last active participant. It publishes and
        // claims its own slice in one critical section, so no worker can
        // observe a half-written task.
        Slice s;
        {
            std::unique_lock<std::mutex> lk(mu_);
            n_ = n;
            grain_ = grain;
            blocks_ = blocks;
            stride_ = stride;
            participants_ = participants;
            body_ctx_ = &body;
            body_fn_ = invoke;
            done_ = 0;
            // Release-order bump: workers acquire-load the id, so every
            // field written above is visible the moment they observe it.
            task_id_.fetch_add(1, std::memory_order_release);
            s = snapshot();
            cv_start_.notify_all();
        }
        run_slice(s, participants - 1);
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_done_.wait(lk, [&] { return done_ >= blocks_; });
            body_ctx_ = nullptr;
            body_fn_ = nullptr;
        }
    }

private:
    // How long a worker spins (watching the atomic task id, burning no
    // syscalls) before parking on the condition variable. Decode's op
    // gaps are microseconds; a couple of milliseconds of spin covers a
    // whole forward pass.
    static constexpr int kSpinMs = 2;

    // One participant's slice of the current task, plus the geometry it was
    // computed from. The caller rewrites n_/grain_/blocks_/stride_ for the
    // NEXT task the moment every participant has reported done, so these
    // fields may only be read while mu_ is held. Reading them unlocked let a
    // participant run a slice built from one task's geometry against another
    // task's body, covering some indices twice and others not at all
    // (test_par_pool_covers_every_block failed ~1 run in 3).
    struct Slice {
        i64 n = 0, grain = 0, blocks = 0, stride = 0, participants = 0;
        void *ctx = nullptr;
        void (*invoke)(void *, i64, i64) = nullptr;
    };

    // Snapshot the published geometry. Called under mu_.
    Slice snapshot() const {
        return Slice{n_, grain_, blocks_, stride_, participants_,
                     body_ctx_, body_fn_};
    }

    // Run blocks [me*stride, min((me+1)*stride, blocks)) of the task
    // described by `s`, then report completion. `s` is a private copy taken
    // under the lock, so a concurrent publish of the next task cannot change
    // the range out from under the body while it runs.
    void run_slice(const Slice &s, i64 me) {
        const i64 b = me * s.stride;
        if (b >= s.blocks) return;
        const i64 e = std::min(b + s.stride, s.blocks);
        for (i64 i = b; i < e; i++)
            s.invoke(s.ctx, i * s.grain, std::min((i + 1) * s.grain, s.n));
        // One completion report per participant per task — staggered by
        // work completion, so the mutex sees almost no contention.
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ += e - b;
            if (done_ >= blocks_) cv_done_.notify_one();
        }
    }

    Pool() {
        const i64 hw = std::max<i64>(1, static_cast<i64>(std::thread::hardware_concurrency()));
        i64 hc = hw;
        bool hc_forced = false;
        if (const char *e = std::getenv("KRK_THREADS")) {
            const i64 v = std::atoll(e);
            if (v > 0) { hc = v; hc_forced = true; }
        }
        topo_ = detect_cpu_topology(hw);
        // WIDTH. The default is the number that was measured on this class of
        // part, and the measurement says the P cluster WITH its SMT siblings --
        // not the physical P count, and not every logical CPU.
        //
        // Qwen3.8-4B-Q4_K_M (2.59 GB, 22-token prompt, -n 32, greedy), one
        // width per round with the starting width rotated each round so the
        // warm-up slot is not one arm's:
        //
        //   width       prefill ms   prefill tok/s   decode tok/s
        //   4              2161.8          14.85           3.45
        //   6              1927.4          16.66           3.85
        //   8  (default)   1507.2          21.29           4.10
        //  10              2048.1          15.63           3.35
        //  14              2125.1          15.06           2.55
        //
        // The shape is the memory-bandwidth curve the same box reports from
        // kraken-membw (512 MiB, median of 9): 8 threads peak at 26.5 GB/s,
        // 14 at 24.8, 18 at 20.4 -- 23% below 8. A decode step is one stream
        // per participant and the P cluster's SMT pairs are what supply the
        // extra memory-level parallelism; 4 threads (one per physical P core)
        // has half the streams and measures 16% slower, which is why the pool
        // does NOT cap to physical cores. An older note here claimed the
        // opposite from a single sequential sweep; that reading was page-cache
        // warm-up (see par_small_width in backend_cpu.cpp, which found the
        // physical cap 20% slower once the A/B was interleaved).
        //
        // So on a hybrid part the slow cluster never joins the pool unless the
        // user asks (KRK_THREADS=<all> or --threads). Where the topology could
        // not be read at all there is no fast cluster to find, and taking EVERY
        // logical CPU is the one choice the curve above rules out for a
        // bandwidth-bound kernel -- so fall back to half the logical count
        // rather than all of it. That is a guard, not a measurement.
        if (!hc_forced) {
            if (topo_.hybrid && topo_.n_fast > 0)
                hc = std::min(hc, topo_.n_fast);
            else if (topo_.order.empty())
                hc = std::min(hc, std::max<i64>(1, (hw + 1) / 2));
        }
        if (hc < 1) hc = 1;
        setup_placement_pinning();
        const i64 nw = std::min<i64>(hc - 1, 63);
        const i64 start = task_id_.load(std::memory_order_acquire);
        for (i64 i = 0; i < nw; i++)
            workers_.emplace_back([this, g = gen_.load(), idx = i, start] {
                if (pinned_) pin_this_thread(worker_cpu(idx));
                worker_loop(g, idx, start);
            });
    }

    // Apply pinning policy to the already-detected topology: auto on hybrid,
    // forced or disabled by KRK_AFFINITY. The caller keeps the fastest core and
    // workers take the cores that follow in fastest-first order.
    void setup_placement_pinning() {
        bool want = topo_.hybrid;
        if (const char *e = std::getenv("KRK_AFFINITY")) want = std::atoll(e) != 0;
        if (!want) return;
        pinned_ = topo_.order.size() > 1 && pin_this_thread(topo_.order[0]);
        topo_.pinned = pinned_;
    }

    i64 worker_cpu(i64 idx) const {
        const i64 n = static_cast<i64>(topo_.order.size());
        if (n == 0) return -1;
        return topo_.order[static_cast<size_t>((idx + 1) % n)];
    }

    ~Pool() {
        quit_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
            cv_start_.notify_all();
        }
        for (std::thread &t : workers_)
            if (t.joinable()) t.join();
    }

    Pool(const Pool &) = delete;
    Pool &operator=(const Pool &) = delete;

    void worker_loop(i64 my_gen, i64 my_idx, i64 seen) {
        // Id of the task this worker has already drained. Waiting for a
        // NEW id (not for body_ != nullptr, which stays set until the
        // caller finishes) is what keeps idle workers asleep instead of
        // spinning on the mutex.
        for (;;) {
            // ---- spin phase ----
            // Watch the atomic task id without touching the mutex: a
            // back-to-back op finds every worker already hot, so the
            // pool costs zero syscalls on the decode hot path.
            bool have = false;
            i64 run_id = 0;
            Slice s;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(kSpinMs);
            for (;;) {
                if (quit_.load(std::memory_order_acquire) ||
                    my_gen != gen_.load(std::memory_order_acquire))
                    return;
                const i64 t = task_id_.load(std::memory_order_acquire);
                if (t > seen) {
                    // Claim the geometry for THIS task id under the lock.
                    // Reading it separately from the id let the two
                    // disagree: a worker could observe id N, then snapshot
                    // the geometry of N+1 after the caller had already
                    // retired it — running a slice twice, or with a body
                    // that had been reset to nullptr.
                    std::lock_guard<std::mutex> lk(mu_);
                    if (task_id_.load(std::memory_order_acquire) > seen) {
                        run_id = task_id_.load(std::memory_order_acquire);
                        s = snapshot();
                        have = s.invoke != nullptr;
                    }
                    if (have) break;
                    // The task was retired before we could claim it (the
                    // caller's fast path can clear body_ without us).
                    // Re-read the id and try again.
                    seen = task_id_.load(std::memory_order_acquire);
                    continue;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    // ---- park phase ----
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_start_.wait(lk, [&] {
                        return quit_.load(std::memory_order_acquire) ||
                               my_gen != gen_.load(std::memory_order_acquire) ||
                               task_id_.load(std::memory_order_acquire) > seen;
                    });
                    if (quit_.load(std::memory_order_acquire) ||
                        my_gen != gen_.load(std::memory_order_acquire))
                        return;
                    // Claim it here too, still under the lock, so the id
                    // and the geometry always come from the same publish.
                    run_id = task_id_.load(std::memory_order_acquire);
                    s = snapshot();
                    have = run_id > seen && s.invoke != nullptr;
                    if (!have) {
                        seen = run_id;
                        lk.unlock();
                        continue;
                    }
                    lk.unlock();
                    break;
                }
                cpu_relax();
            }
            // Workers own slots 0..participants-2; the calling thread owns
            // slot participants-1. A worker outside the active set for this
            // task (active_cap) must skip the slice but still advance `seen`.
            if (s.participants > 1 && my_idx < s.participants - 1)
                run_slice(s, my_idx);
            // Mark the task just drained — NOT whatever task_id_ says now.
            // The caller may publish the next op while we report, and
            // re-reading the id here would swallow that task: its blocks
            // would go uncovered and the caller would wait forever.
            seen = run_id;
        }
    }

    mutable std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::vector<std::thread> workers_;
    CpuTopology topo_;
    bool pinned_ = false;
    void *body_ctx_ = nullptr;
    void (*body_fn_)(void *, i64, i64) = nullptr;
    i64 n_ = 0;
    i64 grain_ = 0;
    i64 blocks_ = 0;
    i64 stride_ = 1;
    i64 participants_ = 0;
    i64 done_ = 0;
    std::atomic<i64> gen_{0};
    std::atomic<i64> task_id_{0};
    std::atomic<bool> quit_{false};
    std::atomic<i64> active_cap_{0}; // 0 = use all participants
};

} // namespace krk

#endif // KRK_PAR_POOL_HPP
