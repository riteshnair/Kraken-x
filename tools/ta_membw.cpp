// Achieved memory read bandwidth on this box, 1 and N threads.
//
// The question it answers: is the Q4_K kernel's ~10 GB/s a *cache/DRAM* wall or
// its own instruction wall? tools/ta_gemv_q4k times the same 10 GB/s on a
// 2.25 MiB matrix (L2-resident) and a 15.75 MiB one, which says the kernel, not
// memory. This probe gives the other side of that comparison: what a pure
// sequential sum over the same buffers actually reaches.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    size_t mb = 2;
    int threads = 1;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--mb") && i + 1 < argc) mb = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    }
    const size_t bytes = mb * 1024ull * 1024ull;
    const size_t n = bytes / sizeof(uint64_t);
    std::vector<uint64_t> buf(n);
    for (size_t i = 0; i < n; i++) buf[i] = i * 6364136223846793005ull + 1;

    // One pass: every thread reads a contiguous slice and sums it, so this is
    // pure load bandwidth with no writes and no sharing.
    auto pass = [&](int nt) {
        std::vector<uint64_t> part(static_cast<size_t>(nt), 0);
        std::vector<std::thread> th;
        const size_t per = n / static_cast<size_t>(nt);
        for (int t = 0; t < nt; t++) {
            th.emplace_back([&, t]() {
                const size_t b = per * static_cast<size_t>(t);
                const size_t e = (t + 1 == nt) ? n : b + per;
                uint64_t s = 0;
                for (size_t i = b; i < e; i++) s += buf[i];
                part[static_cast<size_t>(t)] = s;
            });
        }
        for (auto &x : th) x.join();
        uint64_t s = 0;
        for (auto v : part) s += v;
        return s;
    };

    volatile uint64_t warm = pass(threads);   // page in and warm the caches
    warm = pass(threads);
    (void)warm;

    std::vector<double> t(9);
    for (int i = 0; i < 9; i++) {
        const double a = now_s();
        volatile uint64_t s = pass(threads);
        const double b = now_s();
        (void)s;
        t[static_cast<size_t>(i)] = b - a;
    }
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2];
    const double gbs = static_cast<double>(bytes) / med / 1e9;
    std::printf("%4zu MiB  %2d thread(s)  median %7.3f ms  %6.2f GB/s  best %6.2f GB/s\n",
                mb, threads, med * 1e3, gbs,
                static_cast<double>(bytes) / t.front() / 1e9);
    return 0;
}
