// kv_pages.cpp — bounded reference-counted KV pages and sequence block tables.
#include "krk/kv_pages.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace krk {

bool KvPagePool::init(size_t page_bytes, size_t page_capacity) {
    reset();
    if (page_bytes == 0 || page_capacity > static_cast<size_t>(kInvalidPage) ||
        (page_capacity && page_bytes > std::numeric_limits<size_t>::max() / page_capacity))
        return false;
    try {
        std::vector<std::vector<u8>> pages(page_capacity);
        std::vector<u32> refs(page_capacity, 0);
        std::vector<PageId> free;
        free.reserve(page_capacity);
        for (size_t i = page_capacity; i > 0; --i)
            free.push_back(static_cast<PageId>(i - 1));
        pages_.swap(pages);
        refs_.swap(refs);
        free_.swap(free);
        page_bytes_ = page_bytes;
        return true;
    } catch (const std::bad_alloc &) {
        reset();
        return false;
    } catch (const std::length_error &) {
        reset();
        return false;
    }
}

void KvPagePool::reset() {
    pages_.clear();
    refs_.clear();
    free_.clear();
    page_bytes_ = 0;
    live_pages_ = 0;
    resident_bytes_ = 0;
}

bool KvPagePool::valid(PageId id) const {
    return id != kInvalidPage && static_cast<size_t>(id) < refs_.size() && refs_[id] != 0;
}

bool KvPagePool::allocate(PageId *out) {
    if (!out || free_.empty() || page_bytes_ == 0) return false;
    const PageId id = free_.back();
    std::vector<u8> &page = pages_[id];
    if (page.empty()) {
        try {
            page.resize(page_bytes_, 0);
        } catch (const std::bad_alloc &) {
            return false;
        } catch (const std::length_error &) {
            return false;
        }
        resident_bytes_ += page_bytes_;
    } else {
        std::memset(page.data(), 0, page_bytes_);
    }
    free_.pop_back();
    refs_[id] = 1;
    ++live_pages_;
    *out = id;
    return true;
}

bool KvPagePool::retain(PageId id) {
    if (!valid(id) || refs_[id] == std::numeric_limits<u32>::max()) return false;
    ++refs_[id];
    return true;
}

bool KvPagePool::release(PageId id) {
    if (!valid(id)) return false;
    if (--refs_[id] == 0) {
        std::memset(pages_[id].data(), 0, page_bytes_);
        free_.push_back(id);
        --live_pages_;
    }
    return true;
}

bool KvPagePool::make_writable(PageId *id) {
    if (!id || !valid(*id)) return false;
    if (refs_[*id] == 1) return true;
    PageId fresh = kInvalidPage;
    if (!allocate(&fresh)) return false;
    std::memcpy(mutable_data(fresh), data(*id), page_bytes_);
    const PageId old = *id;
    if (!release(old)) {
        release(fresh);
        return false;
    }
    *id = fresh;
    return true;
}

u8 *KvPagePool::mutable_data(PageId id) {
    return valid(id) && !pages_[id].empty() ? pages_[id].data() : nullptr;
}

const u8 *KvPagePool::data(PageId id) const {
    return valid(id) && !pages_[id].empty() ? pages_[id].data() : nullptr;
}

u32 KvPagePool::ref_count(PageId id) const {
    return valid(id) ? refs_[id] : 0;
}

KvBlockTable::~KvBlockTable() { reset(); }

KvBlockTable::KvBlockTable(KvBlockTable &&other) noexcept
    : pool_(other.pool_), rows_per_page_(other.rows_per_page_),
      blocks_(std::move(other.blocks_)) {
    other.pool_ = nullptr;
    other.rows_per_page_ = 1;
}

KvBlockTable &KvBlockTable::operator=(KvBlockTable &&other) noexcept {
    if (this == &other) return *this;
    reset();
    pool_ = other.pool_;
    rows_per_page_ = other.rows_per_page_;
    blocks_ = std::move(other.blocks_);
    other.pool_ = nullptr;
    other.rows_per_page_ = 1;
    return *this;
}

bool KvBlockTable::bind(KvPagePool *pool, size_t rows_per_page) {
    if (!blocks_.empty() || !pool || rows_per_page == 0) return false;
    pool_ = pool;
    rows_per_page_ = rows_per_page;
    return true;
}

bool KvBlockTable::clone_from(const KvBlockTable &source) {
    if (!source.pool_ || source.rows_per_page_ == 0 ||
        source.blocks_.size() > std::numeric_limits<size_t>::max() /
                                    source.rows_per_page_)
        return false;
    return clone_prefix_from(source, source.blocks_.size() * source.rows_per_page_);
}

bool KvBlockTable::clone_prefix_from(const KvBlockTable &source,
                                     size_t logical_rows) {
    if (this == &source) return logical_rows <= blocks_.size() * rows_per_page_;
    if (!source.pool_ || source.rows_per_page_ == 0) return false;
    if (pool_ && (pool_ != source.pool_ || rows_per_page_ != source.rows_per_page_))
        return false;
    const size_t count = logical_rows == 0
                             ? 0
                             : 1 + (logical_rows - 1) / source.rows_per_page_;
    if (count > source.blocks_.size()) return false;
    std::vector<KvPagePool::PageId> copy;
    try {
        copy.reserve(count);
    } catch (const std::bad_alloc &) {
        return false;
    } catch (const std::length_error &) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const KvPagePool::PageId id = source.blocks_[i];
        if (!source.pool_->retain(id)) {
            for (KvPagePool::PageId held : copy) source.pool_->release(held);
            return false;
        }
        copy.push_back(id);
    }
    reset();
    pool_ = source.pool_;
    rows_per_page_ = source.rows_per_page_;
    blocks_.swap(copy);
    return true;
}

u8 *KvBlockTable::writable_page(size_t logical_position) {
    if (!pool_ || rows_per_page_ == 0) return nullptr;
    const size_t wanted = logical_position / rows_per_page_;
    if (wanted == std::numeric_limits<size_t>::max()) return nullptr;
    if (wanted >= blocks_.size()) {
        const size_t count = wanted + 1 - blocks_.size();
        std::vector<KvPagePool::PageId> added;
        try {
            added.reserve(count);
            blocks_.reserve(wanted + 1);
        } catch (const std::bad_alloc &) {
            return nullptr;
        } catch (const std::length_error &) {
            return nullptr;
        }
        for (size_t i = 0; i < count; ++i) {
            KvPagePool::PageId id = KvPagePool::kInvalidPage;
            if (!pool_->allocate(&id)) {
                for (KvPagePool::PageId allocated : added) pool_->release(allocated);
                return nullptr;
            }
            added.push_back(id);
        }
        blocks_.insert(blocks_.end(), added.begin(), added.end());
    } else if (!pool_->make_writable(&blocks_[wanted])) {
        return nullptr;
    }
    return pool_->mutable_data(blocks_[wanted]);
}

bool KvBlockTable::truncate(size_t logical_rows) {
    if (!pool_ || rows_per_page_ == 0) return false;
    const size_t keep = logical_rows == 0 ? 0 : 1 + (logical_rows - 1) / rows_per_page_;
    if (keep >= blocks_.size()) return true;
    for (size_t i = keep; i < blocks_.size(); ++i)
        if (!pool_->release(blocks_[i])) return false;
    blocks_.resize(keep);
    return true;
}

void KvBlockTable::reset() {
    if (pool_) {
        for (KvPagePool::PageId id : blocks_) pool_->release(id);
    }
    blocks_.clear();
}

KvPagePool::PageId KvBlockTable::block(size_t index) const {
    return index < blocks_.size() ? blocks_[index] : KvPagePool::kInvalidPage;
}

} // namespace krk
