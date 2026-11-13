// ============================================================================
//  moe_cache_replay.cpp -- compare expert-cache configurations offline.
//
//  Reads the TSV that `KRK_TRACE_EXPERTS=1` writes (seq, layer, expert, tier),
//  builds one uniform synthetic expert bank per layer and replays the trace
//  through the real ExpertCache (see include/krk/expert_replay.hpp). Every
//  number it prints is a counter of that class -- it is not a model of the
//  policy, it is the policy, driven in the recorded order.
//
//  Sizes are expressed in SLOTS, not bytes, and one slot is `--expert-kib` of
//  synthetic weight: the policy's decisions depend on budget/ExpertFootprint,
//  so a slot count is the exact quantity and MiB would only add a division that
//  changes per model. `scripts/moe_cache_tune.py` converts MiB to slots with
//  the model's real footprint and turns the counters into a time prediction.
//
//  Usage:
//    kraken-expert-replay --trace expert_trace.tsv --budget-slots 512,1024 \
//        --share 0,0.5,1 --layers 60 --experts 256
//
//  Output is one TSV row per (budget, share) with a header, or `--json`.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "krk/backend.hpp"
#include "krk/expert_cache.hpp"
#include "krk/expert_replay.hpp"

namespace {

struct Args {
    std::string trace;
    std::vector<size_t> budget_slots;
    size_t warm_slots = 0;
    size_t pin_slots = 0;
    std::vector<int> share_permille{1000};
    int layers = 0;
    int experts = 0;
    size_t expert_kib = 16;
    bool json = false;
};

void usage() {
    std::fprintf(stderr,
        "usage: kraken-expert-replay --trace FILE [options]\n"
        "  --trace FILE         routing trace (seq\\tlayer\\texpert\\ttier)\n"
        "  --budget-slots LIST  device tier size in expert slots (comma list)\n"
        "  --warm-slots N       WARM (pageable host) tier in slots (0 = off)\n"
        "  --pin-slots N        pin ring in slots, outside the pool budget\n"
        "  --share LIST         host share of the overflow, 0..1 (comma list)\n"
        "  --layers N           override the layer count the trace implies\n"
        "  --experts N          override the expert count the trace implies\n"
        "  --expert-kib N       synthetic bytes per expert (default 16)\n"
        "  --json               emit JSON instead of TSV\n");
}

bool parse_list(const char *s, std::vector<size_t> *out) {
    out->clear();
    const char *p = s;
    while (*p) {
        char *end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0) return false;
        out->push_back(static_cast<size_t>(v));
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return !out->empty();
}

bool parse_shares(const char *s, std::vector<int> *out) {
    out->clear();
    const char *p = s;
    while (*p) {
        char *end = nullptr;
        const double v = std::strtod(p, &end);
        if (end == p || v < 0.0 || v > 1.0) return false;
        out->push_back(static_cast<int>(v * 1000.0 + 0.5));
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return !out->empty();
}

bool load_trace(const std::string &path, std::vector<krk::ReplayEvent> *out,
                int *n_layer, int *n_expert) {
    FILE *f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    if (!std::fgets(line, sizeof(line), f)) {
        std::fclose(f);
        return false;
    }
    // The header is the contract: a file that does not carry it is not one of
    // ours, and silently reading the first line as data is how a tuner ends up
    // reporting a cache that was never driven.
    if (std::strncmp(line, "seq\tlayer\texpert\ttier", 21) != 0) {
        std::fclose(f);
        return false;
    }
    int max_layer = -1;
    int max_expert = -1;
    while (std::fgets(line, sizeof(line), f)) {
        unsigned seq = 0, tier = 0;
        int layer = 0, expert = 0;
        if (std::sscanf(line, "%u\t%d\t%d\t%u", &seq, &layer, &expert, &tier) != 4)
            continue;
        if (layer < 0 || expert < 0) continue;
        out->push_back(krk::ReplayEvent{layer, expert});
        if (layer > max_layer) max_layer = layer;
        if (expert > max_expert) max_expert = expert;
    }
    std::fclose(f);
    *n_layer = max_layer + 1;
    *n_expert = max_expert + 1;
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        const char *next = i + 1 < argc ? argv[i + 1] : nullptr;
        auto take = [&]() -> const char * {
            if (next) i++;
            return next;
        };
        if (f == "--trace" && next) a.trace = take();
        else if (f == "--budget-slots" && next) {
            if (!parse_list(take(), &a.budget_slots)) { usage(); return 2; }
        } else if (f == "--warm-slots" && next) a.warm_slots = std::strtoul(take(), nullptr, 10);
        else if (f == "--pin-slots" && next) a.pin_slots = std::strtoul(take(), nullptr, 10);
        else if (f == "--share" && next) {
            if (!parse_shares(take(), &a.share_permille)) { usage(); return 2; }
        } else if (f == "--layers" && next) a.layers = std::atoi(take());
        else if (f == "--experts" && next) a.experts = std::atoi(take());
        else if (f == "--expert-kib" && next) a.expert_kib = std::strtoul(take(), nullptr, 10);
        else if (f == "--json") a.json = true;
        else if (f == "--help" || f == "-h") { usage(); return 0; }
        else { usage(); return 2; }
    }
    if (a.trace.empty()) { usage(); return 2; }

    std::vector<krk::ReplayEvent> events;
    int n_layer = 0, n_expert = 0;
    if (!load_trace(a.trace, &events, &n_layer, &n_expert)) {
        std::fprintf(stderr, "kraken-expert-replay: cannot read %s (expected a "
                             "seq/layer/expert/tier TSV from KRK_TRACE_EXPERTS)\n",
                     a.trace.c_str());
        return 1;
    }
    if (a.layers > 0) n_layer = a.layers;
    if (a.experts > 0) n_expert = a.experts;
    if (n_layer <= 0 || n_expert <= 0 || events.empty()) {
        std::fprintf(stderr, "kraken-expert-replay: empty trace\n");
        return 1;
    }

    // ONE bank shared by every layer: the cache only ever reads these bytes, so
    // a per-layer copy would multiply the tool's footprint by the layer count
    // for no fidelity at all -- the policy sees sizes and keys, not contents.
    const size_t per_expert = a.expert_kib * 1024;
    const size_t slice = per_expert / 3 ? per_expert / 3 : 1;
    const size_t bytes = slice * static_cast<size_t>(n_expert);
    std::vector<unsigned char> bank(bytes, 0x3c);
    krk::GgufTensor gate, up, down;
    for (krk::GgufTensor *t : {&gate, &up, &down}) {
        t->type = krk::DType::Q8_0;
        t->n_dims = 3;
        t->ne[0] = 32;
        t->ne[1] = 1;
        t->ne[2] = static_cast<unsigned long long>(n_expert);
        t->n_elements = static_cast<krk::i64>(bytes);
        t->n_bytes = bytes;
        t->data = bank.data();
    }
    krk::ExpertSource src;
    src.gate = &gate;
    src.up = &up;
    src.down = &down;
    src.n_expert = n_expert;
    src.n_embd = 32;
    src.n_ff_exp = 1;
    std::vector<krk::ExpertSource> srcs(static_cast<size_t>(n_layer), src);
    std::vector<const krk::ExpertSource *> src_of(static_cast<size_t>(n_layer));
    for (int l = 0; l < n_layer; l++) src_of[static_cast<size_t>(l)] = &srcs[static_cast<size_t>(l)];

    if (a.budget_slots.empty())
        a.budget_slots.push_back(static_cast<size_t>(n_layer) *
                                 static_cast<size_t>(n_expert));
    const size_t slot_bytes = per_expert;

    krk::Backend *be = krk::make_cpu_backend();
    if (!be) {
        std::fprintf(stderr, "kraken-expert-replay: no CPU backend\n");
        return 1;
    }

    if (!a.json) {
        std::printf("budget_slots\tshare\thits\tpromotions\tloads\thost_takes\t"
                    "evictions\twarm_adm\twarm_rej\tdemotions\tread_kiB\tpromoted_kiB\tresident_slots\t"
                    "pinned_slots\twarm_slots\tsteps\tskipped\n");
    } else {
        std::printf("[\n");
    }
    bool first = true;
    for (size_t slots : a.budget_slots) {
        for (int permille : a.share_permille) {
            krk::ReplayConfig cfg;
            cfg.budget_bytes = slots * slot_bytes;
            cfg.warm_bytes = a.warm_slots * slot_bytes;
            cfg.pin_bytes = a.pin_slots * slot_bytes;
            cfg.share_permille = permille;
            const krk::ReplayResult r = krk::replay_expert_trace(
                be, src_of.data(), n_layer, events.data(), events.size(), cfg);
            if (a.json) {
                std::printf("%s  {\"budget_slots\": %zu, \"share\": %.3f, "
                            "\"hits\": %llu, \"promotions\": %llu, \"loads\": %llu, "
                            "\"host_takes\": %llu, \"evictions\": %llu, "
                            "\"warm_admissions\": %llu, \"warm_rejects\": %llu, "
                            "\"demotions\": %llu, "
                            "\"read_bytes\": %llu, \"promoted_bytes\": %llu, "
                            "\"resident_slots\": %zu, \"pinned_slots\": %zu, "
                            "\"warm_slots\": %zu, \"steps\": %llu, \"skipped\": %llu}",
                            first ? "" : ",\n", slots, permille / 1000.0,
                            (unsigned long long)r.hits,
                            (unsigned long long)r.promotions,
                            (unsigned long long)r.loads,
                            (unsigned long long)r.host_takes,
                            (unsigned long long)r.evictions,
                            (unsigned long long)r.warm_admissions,
                            (unsigned long long)r.warm_rejects,
                            (unsigned long long)r.demotions,
                            (unsigned long long)r.bytes_read,
                            (unsigned long long)r.bytes_promoted,
                            r.resident_slots, r.pinned_slots, r.warm_slots,
                            (unsigned long long)r.steps,
                            (unsigned long long)r.skipped);
            } else {
                std::printf("%zu\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t"
                            "%.0f\t%.0f\t%zu\t%zu\t%zu\t%llu\t%llu\n",
                            slots, permille / 1000.0,
                            (unsigned long long)r.hits,
                            (unsigned long long)r.promotions,
                            (unsigned long long)r.loads,
                            (unsigned long long)r.host_takes,
                            (unsigned long long)r.evictions,
                            (unsigned long long)r.warm_admissions,
                            (unsigned long long)r.warm_rejects,
                            (unsigned long long)r.demotions,
                            static_cast<double>(r.bytes_read) / 1024.0,
                            static_cast<double>(r.bytes_promoted) / 1024.0,
                            r.resident_slots, r.pinned_slots, r.warm_slots,
                            (unsigned long long)r.steps,
                            (unsigned long long)r.skipped);
            }
            first = false;
        }
    }
    if (a.json) std::printf("\n]\n");
    delete be;
    return 0;
}
