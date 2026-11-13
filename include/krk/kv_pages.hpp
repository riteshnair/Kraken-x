// kv_pages.hpp — reference-counted position pages for sequence KV tables.
//
// The pool owns fixed-size, zero-initialized pages. Tables own references to
// pages, and can be forked cheaply with clone_from(); a write detaches a shared
// page before exposing its bytes. This is the storage primitive used by the
// paged-KV engine path; callers remain responsible for translating logical
// positions to page-local offsets in their attention backend.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "krk/common.hpp"

namespace krk {

class KvPagePool {
public:
    using PageId = u32;
    static constexpr PageId kInvalidPage = UINT32_MAX;

    KvPagePool() = default;
    KvPagePool(const KvPagePool &) = delete;
    KvPagePool &operator=(const KvPagePool &) = delete;

    // Allocate a bounded arena. Returns false on overflow or allocation failure
    // and leaves the pool empty. A zero page count is valid but cannot allocate.
    bool init(size_t page_bytes, size_t page_capacity);
    void reset();

    size_t page_bytes() const { return page_bytes_; }
    size_t capacity() const { return refs_.size(); }
    size_t live_pages() const { return live_pages_; }

    bool allocate(PageId *out);
    bool retain(PageId id);
    bool release(PageId id);
    // If shared, allocate and copy a private page, release the old reference,
    // and replace *id. Unique pages are left untouched.
    bool make_writable(PageId *id);

    const u8 *data(PageId id) const;
    u32 ref_count(PageId id) const;

private:
    friend class KvBlockTable;
    u8 *mutable_data(PageId id);
    bool valid(PageId id) const;

    size_t page_bytes_ = 0;
    size_t live_pages_ = 0;
    std::vector<u8> bytes_;
    std::vector<u32> refs_;
    std::vector<PageId> free_;
};

// A sequence's logical block table. Cloning a table increments page refs but
// does not copy KV payload. writable_page() provides page-level COW; truncate
// and reset release references so slots can be reused without leaks.
class KvBlockTable {
public:
    explicit KvBlockTable(KvPagePool *pool = nullptr, size_t rows_per_page = 1)
        : pool_(pool), rows_per_page_(rows_per_page) {}
    ~KvBlockTable();
    KvBlockTable(const KvBlockTable &) = delete;
    KvBlockTable &operator=(const KvBlockTable &) = delete;
    KvBlockTable(KvBlockTable &&other) noexcept;
    KvBlockTable &operator=(KvBlockTable &&other) noexcept;

    bool bind(KvPagePool *pool, size_t rows_per_page);
    bool clone_from(const KvBlockTable &source);
    // Returns the full mutable page containing logical_position, detaching it
    // first when shared. The caller owns the logical-row-to-byte offset math.
    u8 *writable_page(size_t logical_position);
    bool truncate(size_t logical_rows);
    void reset();

    size_t block_count() const { return blocks_.size(); }
    KvPagePool::PageId block(size_t index) const;
    KvPagePool *pool() const { return pool_; }

private:
    KvPagePool *pool_ = nullptr;
    size_t rows_per_page_ = 1;
    std::vector<KvPagePool::PageId> blocks_;
};

} // namespace krk
