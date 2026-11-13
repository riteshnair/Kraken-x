// ============================================================================
//  expert_replay.cpp -- see include/krk/expert_replay.hpp for the contract.
//
//  The replay is deliberately thin: every residency decision below is made by
//  ExpertCache, and this file only supplies the event order, the leases and the
//  host-arm share the engine would have used. Anything more would be a second
//  policy, and a second policy is what the tuner must never have.
// ============================================================================
#include "krk/expert_replay.hpp"

#include <utility>
#include <vector>

namespace krk {

u64 replay_steps(const ReplayEvent *events, size_t n) {
    if (!events || n == 0) return 0;
    u64 steps = 1;
    for (size_t i = 1; i < n; i++) {
        // A forward pass walks the layers in order, so the index going
        // backwards is the only signal a trace carries for "another token".
        // It needs no top-k: a layer that is not MoE never appears.
        if (events[i].layer < events[i - 1].layer) steps++;
    }
    return steps;
}

ReplayResult replay_expert_trace(Backend *be, const ExpertSource *const *src_of,
                                 i32 n_layer, const ReplayEvent *events, size_t n,
                                 const ReplayConfig &cfg) {
    ReplayResult r;
    if (!be || !src_of || !events || n == 0) return r;

    ExpertCache cache;
    cache.configure(be, cfg.budget_bytes, cfg.warm_bytes, /*slot_budget=*/0);
    // After configure(), which resets the tier it decorates.
    if (cfg.pin_bytes) cache.set_pin_budget(cfg.pin_bytes);

    r.steps = replay_steps(events, n);

    std::vector<std::pair<i32, i32>> leased;
    i32 cur_layer = -1;
    for (size_t i = 0; i < n; i++) {
        const i32 layer = events[i].layer;
        const i32 expert = events[i].expert;
        if (layer < 0 || layer >= n_layer) {
            r.skipped++;
            continue;
        }
        const ExpertSource *src = src_of[layer];
        if (!src || !src->present()) {
            r.skipped++;
            continue;
        }
        // The engine holds a layer's experts until that layer's launches have
        // retired. Releasing at the first event of the next layer is the same
        // window, and it matters on a small budget: without it a replay promotes
        // an expert that the run would have kept and then re-reads it.
        if (cfg.lease_layers && layer != cur_layer) {
            for (const auto &le : leased)
                cache.set_lease(le.first, le.second, false);
            leased.clear();
            cur_layer = layer;
        }
        // The host arm takes a share of the OVERFLOW only (see
        // expert_split_share): a miss the device tier can still afford is
        // promoted exactly as it is without --hybrid-experts.
        if (!cache.in_vram(layer, expert) && cache.full_for(*src) &&
            expert_split_share(layer, expert, cfg.share_permille)) {
            r.host_takes++;
            continue;
        }
        r.acquire++;
        const ResidentExpert *re = cache.acquire(*src, layer, expert);
        if (!re) {
            // A refused allocation is not a miss: the engine fails the
            // generation here, so a replay that counted it as traffic would
            // predict a run that never finished.
            r.skipped++;
            continue;
        }
        if (cfg.lease_layers) {
            cache.set_lease(layer, expert, true);
            leased.emplace_back(layer, expert);
        }
    }

    r.hits = cache.hits();
    r.promotions = cache.host_hits();
    r.loads = cache.loads();
    r.evictions = cache.evictions();
    r.warm_admissions = cache.warm_admissions();
    r.warm_rejects = cache.warm_rejects();
    r.demotions = cache.demotions();
    r.bytes_read = cache.bytes_loaded();
    r.bytes_promoted = cache.bytes_promoted();
    r.resident_bytes = static_cast<u64>(cache.resident_bytes());
    r.resident_slots = cache.resident_slots();
    r.warm_slots = cache.warm_slots();
    r.pinned_slots = cache.pinned_slots();
    // Release the cells this replay promoted. `ExpertCache` does not release its
    // resident slots on destruction (a cache may outlive a `configure()` that
    // re-tiers it), so a caller that only drops the object hands the backend
    // nothing back: LeakSanitizer sees the pooled blocks as still live, which is
    // exactly what they are. Clearing here is also what makes the next replay in
    // the same process start from the budget it asked for rather than from the
    // previous trace's residency.
    cache.clear();
    return r;
}

} // namespace krk
