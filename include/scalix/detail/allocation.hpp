//------------------------------------------------------------------------------
// BSD 3-Clause License
//
// Copyright (c) 2023 Jack Myers
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------

#pragma once

#include <scalix/defines.hpp>
#include <scalix/detail/page_handle.hpp>
#include <scalix/pointers.hpp>
#include <unordered_map>
#include <vector>

namespace sclx::detail {

enum class reuse_pages : std::uint8_t { if_possible, no };

template<page_size_t PageSize>
class allocation_handle {
  public:
    using page_handle = page_handle<page_handle_type::strong, PageSize>;

    allocation_handle()                                            = default;
    allocation_handle(const allocation_handle&)                    = default;
    allocation_handle(allocation_handle&&)                         = default;
    auto operator=(const allocation_handle&) -> allocation_handle& = default;
    auto operator=(allocation_handle&&) -> allocation_handle&      = default;

    [[nodiscard]] virtual auto device_id() const -> device_id_t = 0;

    virtual ~allocation_handle() = default;
};

template<page_size_t PageSize>
class device_allocation_anchor;

template<page_size_t PageSize>
struct access_anchor_creator_struct;

template<page_size_t PageSize>
class device_allocation_anchor {
  public:
    using page_handle       = page_handle<page_handle_type::strong, PageSize>;
    using allocation_handle = allocation_handle<PageSize>;

    device_allocation_anchor()                                = default;
    device_allocation_anchor(const device_allocation_anchor&) = default;
    device_allocation_anchor(device_allocation_anchor&&)      = default;
    auto operator=(const device_allocation_anchor&)
        -> device_allocation_anchor& = default;
    auto operator=(device_allocation_anchor&&)
        -> device_allocation_anchor& = default;

    [[nodiscard]] auto device_id() const -> device_id_t { return device_id_; }

    [[nodiscard]] auto pages() const -> const std::vector<page_handle>& {
        return pages_;
    }

    friend struct access_anchor_creator_struct<PageSize>;

    ~device_allocation_anchor() = default;

  private:
    explicit device_allocation_anchor(
        device_id_t device_id,
        std::vector<page_handle> pages
    )
        : device_id_(device_id),
          pages_(std::move(pages)) {}

    device_id_t device_id_;
    std::vector<page_handle> pages_;
};

template<page_size_t PageSize>
struct access_anchor_creator_struct {
    static auto create(
        device_id_t device_id,
        const std::vector<page_handle<page_handle_type::strong, PageSize>>&
            pages
    ) -> device_allocation_anchor<PageSize> {
        return device_allocation_anchor<PageSize>(device_id, pages);
    }
};

template<page_size_t PageSize>
auto make_access_anchor(
    device_id_t device_id,
    const std::vector<page_handle<page_handle_type::strong, PageSize>>& pages
) -> device_allocation_anchor<PageSize> {
    return access_anchor_creator_struct<PageSize>::create(device_id, pages);
}

template<
    class T,
    template<class, reuse_pages, page_size_t>
    class AllocationType,
    reuse_pages ReusePagesFlag,
    page_size_t PageSize,
    class... Args>
concept AllocationConcept = requires(
    AllocationType<T, ReusePagesFlag, PageSize> alloc,
    Args&&... args
) {
    {
        AllocationType<T, ReusePagesFlag, PageSize>(
            device_id_t(),
            std::declval<const std::vector<page_index_t>&>(),
            std::declval<const std::vector<
                page_handle<page_handle_type::weak, PageSize>>&>(),
            std::forward<Args>(args)...
        )
    };
    {
        AllocationType<T, ReusePagesFlag, PageSize>(
            device_id_t(),
            page_index_t(),
            page_index_t(),
            std::declval<const std::vector<
                page_handle<page_handle_type::weak, PageSize>>&>(),
            std::forward<Args>(args)...
        )
    };
    {
        static_cast<const AllocationType<T, ReusePagesFlag, PageSize>&>(alloc)
            .anchor()
    } -> std::same_as<const device_allocation_anchor<PageSize>&>;
};

template<class T, page_size_t PageSize = default_page_size>
class allocation_factory {
  public:
    using strong_page_handle = page_handle<page_handle_type::strong, PageSize>;
    using weak_page_handle   = page_handle<page_handle_type::weak, PageSize>;
    using strong_page_vector = std::vector<strong_page_handle>;
    using weak_page_vector   = std::vector<weak_page_handle>;

    template<
        template<class, reuse_pages, page_size_t>
        class AllocationType,
        reuse_pages ReusePagesFlag,
        class... Args>
        requires AllocationConcept<
            T,
            AllocationType,
            ReusePagesFlag,
            PageSize,
            Args...>
    void allocate_pages(
        device_id_t device_id,
        const std::vector<page_index_t>& indices,
        const weak_page_vector& weak_pages,
        Args&&... args
    ) {
        using allocation = AllocationType<T, ReusePagesFlag, PageSize>;
        allocation
        alloc(device_id, indices, weak_pages, std::forward<Args>(args)...);
        add_pages_from_anchor(alloc.anchor());
    }

    template<
        template<class, reuse_pages, page_size_t>
        class AllocationType,
        reuse_pages ReusePagesFlag,
        class... Args>
        requires AllocationConcept<
            T,
            AllocationType,
            ReusePagesFlag,
            PageSize,
            Args...>
    void allocate_pages(
        device_id_t device_id,
        page_index_t first,
        page_index_t last,
        const weak_page_vector& weak_pages,
        Args&&... args
    ) {
        using allocation = AllocationType<T, ReusePagesFlag, PageSize>;
        allocation
        alloc(device_id, first, last, weak_pages, std::forward<Args>(args)...);
        add_pages_from_anchor(alloc.anchor());
    }

    void add_pages_from_anchor(const device_allocation_anchor<PageSize>& anchor
    ) {
        auto& device_pages = pages_[anchor.device_id()];
        std::transform(
            anchor.pages().begin(),
            anchor.pages().end(),
            std::back_inserter(device_pages),
            [](auto& page) { return weak_page_handle(page); }
        );
        anchors_.push_back(anchor);
    }

    auto pages(device_id_t device_id) -> weak_page_vector& {
        return pages_[device_id];
    }

  private:
    std::unordered_map<device_id_t, weak_page_vector> pages_;
    std::vector<device_allocation_anchor<PageSize>> anchors_;
};

}  // namespace sclx::detail
