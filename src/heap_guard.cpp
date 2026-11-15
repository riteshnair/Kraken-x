// heap_guard.cpp -- debug-only front-canary + freed-block quarantine.
//
// Built only with KRK_HEAP_GUARD=ON. Every operator new allocation carries a
// front canary covering the whole alignment gap and a trailing canary; delete
// validates both, then poisons the payload and QUARANTINES the block instead of
// returning it to the heap. A process-exit scan reports any quarantined payload
// that changed after it was freed -- that is a write-after-free, the one class
// the tail-canary cannot see and the reason the exit-time heap corruption
// escapes a normal guard. Not part of a normal build.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#endif

namespace {
constexpr size_t kGuard = 16;
constexpr size_t kMinGap = 256;
constexpr size_t kMagic = 0x4B524B4B524B4B03ull;
constexpr unsigned char kCanary = 0xAB;
constexpr unsigned char kPoison = 0xCD;
constexpr size_t kMaxFrames = 10;

struct Head {
    size_t magic;
    size_t size;
    size_t gap;
    void *raw;
    unsigned short nframes;
    void *frames[kMaxFrames];
};

struct Footer {
    size_t magic;
    size_t gap;
};

// Intrusive list whose nodes come from HeapAlloc, not operator new: a
// std::vector here would reallocate through the guarded new and deadlock on the
// quarantine lock it already holds.
struct QNode {
    QNode *next;
    void *payload;
    size_t size;
    unsigned short nframes;
    void *frames[kMaxFrames];
};

QNode *g_head = nullptr;
size_t g_bytes = 0;
constexpr size_t kQuarantineCap = static_cast<size_t>(768) << 20;

std::mutex &quarantine_mu() {
    static std::mutex m;
    return m;
}

inline void *node_alloc(size_t n) {
#if defined(_WIN32)
    return HeapAlloc(GetProcessHeap(), 0, n);
#else
    return std::malloc(n);
#endif
}
inline void node_free(void *p) {
#if defined(_WIN32)
    HeapFree(GetProcessHeap(), 0, p);
#else
    std::free(p);
#endif
}

inline size_t pow2(size_t n) {
    size_t p = 16;
    while (p < n) p <<= 1;
    return p;
}

inline unsigned char *raw_alloc(size_t total, size_t align) {
#if defined(_WIN32)
    return static_cast<unsigned char *>(_aligned_malloc(total, align));
#else
    void *p = nullptr;
    if (posix_memalign(&p, align, total) != 0) return nullptr;
    return static_cast<unsigned char *>(p);
#endif
}

inline void raw_free(void *p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void fill(unsigned char *p, size_t n, unsigned char v) { std::memset(p, v, n); }
bool intact(const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (p[i] != kCanary) return false;
    return true;
}

void dump(const char *what, size_t size, unsigned short nf, void *const *fr) {
    std::fprintf(stderr, "[heap-guard] %s size=%zu\n", what, size);
    for (unsigned short i = 0; i < nf; i++)
        std::fprintf(stderr, "[heap-guard]   %p\n", fr[i]);
    std::fflush(stderr);
}

void *guarded_alloc(size_t n, size_t align) {
    if (align < kMinGap) align = kMinGap;
    const size_t gap = pow2(align);
    const size_t total = gap + (n ? n : 1) + kGuard;
    unsigned char *raw = raw_alloc(total, gap);
    if (!raw) return nullptr;
    Head *h = reinterpret_cast<Head *>(raw);
    h->magic = kMagic;
    h->size = n;
    h->gap = gap;
    h->raw = raw;
#if defined(_WIN32)
    h->nframes = static_cast<unsigned short>(
        CaptureStackBackTrace(1, kMaxFrames, h->frames, nullptr));
#else
    h->nframes = 0;
#endif
    unsigned char *payload = raw + gap;
    Footer *f = reinterpret_cast<Footer *>(payload) - 1;
    f->magic = kMagic;
    f->gap = gap;
    fill(reinterpret_cast<unsigned char *>(raw) + sizeof(Head),
         gap - sizeof(Head) - sizeof(Footer), kCanary);
    fill(payload, n, kPoison);
    fill(payload + n, kGuard, kCanary);
    return payload;
}

void guarded_free(void *p) {
    if (!p) return;
    unsigned char *payload = static_cast<unsigned char *>(p);
    Footer *f = reinterpret_cast<Footer *>(payload) - 1;
    if (f->magic != kMagic) {
        std::fprintf(stderr, "[heap-guard] bad footer / double free ptr=%p\n", p);
        std::fflush(stderr);
        std::abort();
    }
    unsigned char *raw = payload - f->gap;
    Head *h = reinterpret_cast<Head *>(raw);
    if (h->magic != kMagic) {
        std::fprintf(stderr, "[heap-guard] bad header ptr=%p\n", p);
        std::fflush(stderr);
        std::abort();
    }
    const bool front = intact(raw + sizeof(Head),
                              f->gap - sizeof(Head) - sizeof(Footer));
    const bool back = intact(payload + h->size, kGuard);
    if (!front || !back) {
        dump(front ? "trailing canary" : "front canary", h->size, h->nframes,
             h->frames);
        std::abort();
    }
    // Poison and quarantine: a write after this free changes a byte away from
    // kPoison, and the exit scan catches it.
    fill(payload, h->size, kPoison);
    {
        std::lock_guard<std::mutex> lk(quarantine_mu());
        if (g_bytes + h->size > kQuarantineCap) {
            f->magic = 0;
            h->magic = 0;
            raw_free(h->raw);
            return;
        }
        QNode *n = static_cast<QNode *>(node_alloc(sizeof(QNode)));
        if (!n) {
            f->magic = 0;
            h->magic = 0;
            raw_free(h->raw);
            return;
        }
        n->payload = payload;
        n->size = h->size;
        n->nframes = h->nframes;
        std::memcpy(n->frames, h->frames, sizeof(n->frames));
        n->next = g_head;
        g_head = n;
        g_bytes += h->size;
        f->magic = 0;
        h->magic = 0;
    }
    // Deliberately NOT raw_free: the block stays mapped so a later write is
    // observable and cannot corrupt another allocation's metadata.
}

} // namespace

void *operator new(size_t n) { return guarded_alloc(n, kMinGap); }
void *operator new[](size_t n) { return guarded_alloc(n, kMinGap); }
void *operator new(size_t n, std::align_val_t a) {
    return guarded_alloc(n, static_cast<size_t>(a));
}
void *operator new[](size_t n, std::align_val_t a) {
    return guarded_alloc(n, static_cast<size_t>(a));
}
void operator delete(void *p) noexcept { guarded_free(p); }
void operator delete[](void *p) noexcept { guarded_free(p); }
void operator delete(void *p, size_t) noexcept { guarded_free(p); }
void operator delete[](void *p, size_t) noexcept { guarded_free(p); }
void operator delete(void *p, std::align_val_t) noexcept { guarded_free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { guarded_free(p); }
void operator delete(void *p, size_t, std::align_val_t) noexcept { guarded_free(p); }
void operator delete[](void *p, size_t, std::align_val_t) noexcept { guarded_free(p); }

#if defined(_WIN32)
// Runs at process exit after thread-local and static destructors. Any
// quarantined payload whose poison changed was written after free.
__attribute__((destructor)) void krk_heap_guard_check() {
    size_t n = 0;
    for (QNode *q = g_head; q; q = q->next) {
        const unsigned char *p = static_cast<const unsigned char *>(q->payload);
        bool clean = true;
        for (size_t i = 0; i < q->size; i++)
            if (p[i] != kPoison) {
                clean = false;
                break;
            }
        if (!clean) {
            dump("write-after-free", q->size, q->nframes, q->frames);
            std::abort();
        }
        n++;
    }
    std::fprintf(stderr, "[heap-guard] quarantine clean: %zu blocks\n", n);
    std::fflush(stderr);
}
#endif