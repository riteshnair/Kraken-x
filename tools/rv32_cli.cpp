// Run a flat RV32 binary in Kraken's exact engine-native coprocessor.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "krk/rv32.hpp"

using namespace krk;

static void usage() {
    std::fprintf(stderr,
        "usage: kraken-rv32 PROGRAM.bin [options]\n"
        "  --entry N        load and entry address (default 0)\n"
        "  --memory N       byte RAM size (default 65536)\n"
        "  --max-steps N    instruction limit (default 1000000)\n"
        "  --weights FILE   int4 codes, one byte per code; value = code - 8\n"
        "  --trace FILE     write one TSV record per committed instruction\n"
        "\nECALL a7=1 writes a0 as one output byte; a7=93 exits with a0.\n");
}

static bool read_file(const std::string &path, std::vector<u8> *out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return in.good() || in.eof();
}

static bool number(const char *text, u64 *out) {
    if (!text || !*text || *text == '-') return false;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 0);
    if (!end || *end) return false;
    *out = static_cast<u64>(value);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    std::string program, weights, trace;
    u64 entry = 0, memory = Rv32::kDefaultMemoryBytes, steps = 1000000;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&](u64 *value) -> bool {
            return i + 1 < argc && number(argv[++i], value);
        };
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        else if (arg == "--entry") { if (!next(&entry)) { usage(); return 2; } }
        else if (arg == "--memory") { if (!next(&memory)) { usage(); return 2; } }
        else if (arg == "--max-steps") { if (!next(&steps)) { usage(); return 2; } }
        else if (arg == "--weights" && i + 1 < argc) weights = argv[++i];
        else if (arg == "--trace" && i + 1 < argc) trace = argv[++i];
        else if (!arg.empty() && arg[0] != '-' && program.empty()) program = arg;
        else { usage(); return 2; }
    }
    if (program.empty() || memory == 0 || memory > 0xffffffffu ||
        entry > memory - 4) {
        usage();
        return 2;
    }
    std::vector<u8> bytes;
    if (!read_file(program, &bytes)) {
        std::fprintf(stderr, "kraken-rv32: cannot read %s\n", program.c_str());
        return 2;
    }
    if (bytes.empty() || (bytes.size() & 3) != 0) {
        std::fprintf(stderr, "kraken-rv32: flat program must be a nonempty multiple of 4 bytes\n");
        return 2;
    }
    Rv32 machine(static_cast<u32>(memory));
    std::string err;
    if (!machine.load(static_cast<u32>(entry), bytes.data(), bytes.size(), &err)) {
        std::fprintf(stderr, "kraken-rv32: %s\n", err.c_str());
        return 2;
    }
    machine.reset(static_cast<u32>(entry));
    if (!weights.empty()) {
        std::vector<u8> codes;
        if (!read_file(weights, &codes)) {
            std::fprintf(stderr, "kraken-rv32: cannot read %s\n", weights.c_str());
            return 2;
        }
        machine.set_tensor_weights(std::move(codes));
    }
    machine.set_tracing(!trace.empty());
    const bool ok = machine.run(steps);
    if (!trace.empty()) {
        std::ofstream out(trace);
        out << "instruction\tpc\tword\tnext_pc\trd\trd_before\trd_after\t"
               "memory_address\tmemory_bytes\tmemory_before\tmemory_after\n";
        char line[512];
        for (const Rv32TraceEvent &e : machine.trace()) {
            std::snprintf(line, sizeof(line),
                          "%llu\t%08x\t%08x\t%08x\t%u\t%08x\t%08x\t%08x\t%u\t%08x\t%08x\n",
                          static_cast<unsigned long long>(e.instruction), e.pc, e.word,
                          e.next_pc, e.rd, e.rd_before, e.rd_after, e.memory_address,
                          e.memory_bytes, e.memory_before, e.memory_after);
            out << line;
        }
        if (!out) {
            std::fprintf(stderr, "kraken-rv32: cannot write trace %s\n", trace.c_str());
            return 2;
        }
    }
    if (!machine.output().empty())
        std::fwrite(machine.output().data(), 1, machine.output().size(), stdout);
    std::fprintf(stderr,
                 "kraken-rv32: %s instructions=%llu exit=%u pc=%08x trace=%016llx state=%016llx\n",
                 ok ? "ok" : rv32_fault_name(machine.fault()),
                 static_cast<unsigned long long>(machine.instructions()), machine.exit_code(),
                 machine.fault_pc(), static_cast<unsigned long long>(machine.trace_hash()),
                 static_cast<unsigned long long>(machine.state_hash()));
    return ok && machine.halted() && machine.fault() == Rv32Fault::None ? 0 : 1;
}
