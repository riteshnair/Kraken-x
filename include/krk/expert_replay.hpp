// ============================================================================
//  expert_replay.hpp -- the expert cache, replayed from a routing trace.
//
//  Deciding an expert-cache size, a pin budget or a host share today costs a
//  full model load per arm: `scripts/ab_slots.sh`, `ab_pin_ring.sh` and
//  `ab_prefetch.sh` each start a process, warm it and generate, and the answer
//  is two tok/s numbers 3% apart. The routing that decides those numbers was
//  already recorded -- `KRK_TRACE_EXPERTS=1` writes one line per acquire to
//  `expert_trace.tsv` -- so the same question can be answered offline against
//  the SAME ExpertCache the run used.
//
//  That is the whole point of this file existing rather than a Python model of
//  the policy: a second implementation of LFU + aging + the pin ring + the WARM
//  tier would drift from the first one, silently, and every number the tuner
//  printed would describe a cache that is not the one in this tree. Here the
//  replay drives the real class, so a policy change changes the prediction in
//  the same commit.
//
//  What is modelled, because it changes residency:
//    * the acquire order of the trace, per (layer, expert);
//    * the byte budget and the WARM budget, through the cache's own accounting;
//    * leases, one layer at a time, exactly as the engine holds them;
//    * the host arm's share of the OVERFLOW (see expert_split_share): those
//      acquires are counted and skipped, which is what the engine does.
//
//  What is NOT modelled, and why the predicted token time is a model too:
//    * staging, lookahead and prefetch are advisory paths whose effect on
//      residency is measured by the counters they move, not by event order;
//    * a batched forward resolves a whole layer's routed set before it launches
//      anything, so its acquire order is the layer's, not the token's -- a trace
//      taken from a batched run already carries that order, and one taken from a
//      decode run does not;
//    * the time itself: this file counts events and bytes. Converting those to
//      milliseconds needs the machine's read, promotion and host-compute rates,
//      which is the caller's business (`scripts/moe_cache_tune.py`).
// ============================================================================
#ifndef KRK_EXPERT_REPLAY_HPP
#define KRK_EXPERT_REPLAY_HPP

#include <cstddef>

#include "krk/backend.hpp"
#include "krk/expert_cache.hpp"

namespace krk {

// One acquire, in the order the run issued it. The trace's `tier` column is
// deliberately not carried: a replay that trusted it would be replaying the
// answers of the run it is supposed to re-derive.
struct ReplayEvent {
    i32 layer = 0;
    i32 expert = 0;
};

struct ReplayConfig {
    // Device tier. 0 means "one slot": the cache always allows one, which is
    // what a run with no explicit budget and a card that fits nothing gets.
    size_t budget_bytes = 0;
    // WARM (pageable host) tier. 0 disables it, which is the pre-tier behaviour.
    size_t warm_bytes = 0;
    // Pin ring, charged outside the pool budget. 0 leaves every pin inside it.
    size_t pin_bytes = 0;
    // Share of the device tier's OVERFLOW the host arm computes, in permille:
    // 0 promotes every overflow expert, 1000 (the default) computes all of them
    // on the host, matching `--hybrid-frac`.
    int share_permille = 1000;
    // Hold each layer's experts leased until the layer changes. The engine does
    // this for a multi-expert consumer; a replay that skipped it would evict
    // inside a layer the run could not.
    bool lease_layers = true;
};

struct ReplayResult {
    // Acquires the cache was asked for, and how they were served.
    u64 acquire = 0;
    u64 hits = 0;      // device-resident
    u64 promotions = 0; // WARM -> VRAM
    u64 loads = 0;      // file/mapping read
    u64 host_takes = 0; // computed on the host, never acquired
    u64 skipped = 0;    // layer not in the trace's source table
    u64 evictions = 0;
    // WARM, which is where a "why did the replay not promote anything" question
    // is answered: admissions that landed, and the ones the tier refused.
    u64 warm_admissions = 0;
    u64 warm_rejects = 0;
    u64 demotions = 0;
    // Bytes those events moved, and what stayed resident at the end.
    u64 bytes_read = 0;
    u64 bytes_promoted = 0;
    u64 resident_bytes = 0;
    size_t resident_slots = 0;
    size_t warm_slots = 0;
    size_t pinned_slots = 0;
    // Forward steps in the trace: a decrease in the layer index means the run
    // started another pass, i.e. another token or chunk.
    u64 steps = 0;
};

// How many forward passes `events` contains, i.e. one plus the number of times
// the layer index goes backwards. 0 for an empty trace.
u64 replay_steps(const ReplayEvent *events, size_t n);

// Drives a fresh ExpertCache over `events`. `src_of[layer]` is that layer's
// expert source (null for a dense layer, whose events are counted as skipped).
// The cache is configured here and destroyed before returning, so the caller's
// only state is the result. Returns zeroed counters when a required argument is
// null.
ReplayResult replay_expert_trace(Backend *be, const ExpertSource *const *src_of,
                                 i32 n_layer, const ReplayEvent *events, size_t n,
                                 const ReplayConfig &cfg);

} // namespace krk

#endif // KRK_EXPERT_REPLAY_HPP
