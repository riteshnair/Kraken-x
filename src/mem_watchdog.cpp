// mem_watchdog.cpp -- debug-only range check for large memory copies.
//
// Built only with KRK_MEM_WATCHDOG=ON, on a build that also passes
// -fno-builtin-memcpy/-memset/-memmove so calls reach these definitions. A copy
// whose destination or source is not in a committed region is the wild-pointer
// write ASan caught in the batch path; this aborts at the call site and prints
// the return address, which resolves to a line with the DWARF in the binary.
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
// winnt.h maps these to memcpy/memset macros; use the real ntdll exports so
// the watchdog's own definitions do not recurse into themselves.
#undef RtlMoveMemory
#undef RtlFillMemory
#undef CopyMemory
#undef FillMemory

extern "C" {
NTSYSAPI void NTAPI RtlMoveMemory(void *Destination, const void *Source,
                                  SIZE_T Length);
NTSYSAPI void NTAPI RtlFillMemory(void *Destination, SIZE_T Length,
                                  UCHAR Fill);
}

namespace {
constexpr size_t kCheckMin = 32; // large copies only: cheap enough to keep on

bool range_ok(const void *p, size_t n) {
    if (!p) return n == 0;
    const unsigned char *cur = static_cast<const unsigned char *>(p);
    const unsigned char *end = cur + n;
    if (end < cur) return false; // pointer+size overflow
    while (cur < end) {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(cur, &mbi, sizeof mbi) == 0) return false;
        if (mbi.State != MEM_COMMIT) return false;
        if (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return false;
        const unsigned char *next =
            static_cast<const unsigned char *>(mbi.BaseAddress) + mbi.RegionSize;
        if (next <= cur) return false;
        cur = next;
    }
    return true;
}

[[noreturn]] void report(const char *op, const void *d, const void *s,
                         size_t n) {
    void *caller = __builtin_return_address(0);
    std::fprintf(stderr,
                 "[mem-watchdog] %s out of range: dst=%p src=%p n=%zu caller=%p\n",
                 op, d, s, n, caller);
    std::fflush(stderr);
    std::abort();
}
} // namespace

extern "C" void *memcpy(void *d, const void *s, size_t n) {
    if (n >= kCheckMin && (!range_ok(d, n) || !range_ok(s, n)))
        report("memcpy", d, s, n);
    RtlMoveMemory(d, s, n);
    return d;
}

extern "C" void *memmove(void *d, const void *s, size_t n) {
    if (n >= kCheckMin && (!range_ok(d, n) || !range_ok(s, n)))
        report("memmove", d, s, n);
    RtlMoveMemory(d, s, n);
    return d;
}

extern "C" void *memset(void *d, int v, size_t n) {
    if (n >= kCheckMin && !range_ok(d, n)) report("memset", d, nullptr, n);
    RtlFillMemory(d, n, static_cast<UCHAR>(v));
    return d;
}
#endif
