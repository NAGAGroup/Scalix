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

enum class reuse_pages { if_possible, no };

class generic_allocation_handle {
  public:
    generic_allocation_handle()                                 = default;
    generic_allocation_handle(const generic_allocation_handle&) = default;
    generic_allocation_handle(generic_allocation_handle&&)      = default;
    auto operator=(const generic_allocation_handle&)
        -> generic_allocation_handle& = default;
    auto operator=(generic_allocation_handle&&)
        -> generic_allocation_handle&    = default;
    virtual ~generic_allocation_handle() = default;
};

template<page_size_t PageSize>
class allocation_handle : public generic_allocation_handle {
  public:
    using page_handle = page_handle<page_handle_type::strong, PageSize>;
    virtual auto pages() const -> const std::vector<page_handle>& = 0;
};

class single_device_allocation_anchor;

template<class AllocationHandle>
auto make_access_anchor(device_id_t device_id, AllocationHandle&& handle)
    -> single_device_allocation_anchor;

class single_device_allocation_anchor {
  public:
    template<class AllocationHandle>
    friend auto
    make_access_anchor(device_id_t device_id, AllocationHandle&& handle)
        -> single_device_allocation_anchor {
        return single_device_allocation_anchor(
            device_id,
            std::forward<AllocationHandle>(handle)
        );
    }

    template<page_size_t PageSize>
    [[nodiscard]] auto handle() const -> const allocation_handle<PageSize>& {
        return *std::static_pointer_cast<allocation_handle<PageSize>>(handle_);
    }

    [[nodiscard]] auto device_id() const -> device_id_t { return device_id_; }

    single_device_allocation_anchor() = default;
    single_device_allocation_anchor(const single_device_allocation_anchor&)
        = default;
    single_device_allocation_anchor(single_device_allocation_anchor&&)
        = default;
    auto operator=(const single_device_allocation_anchor&)
        -> single_device_allocation_anchor& = default;
    auto operator=(single_device_allocation_anchor&&)
        -> single_device_allocation_anchor& = default;

    ~single_device_allocation_anchor() = default;

  private:
    template<class AllocationHandle>
    single_device_allocation_anchor(
        device_id_t device_id,
        AllocationHandle&& handle
    )
        : device_id_(device_id),
          handle_(std::make_shared<AllocationHandle>(
              std::forward<AllocationHandle>(handle)
          )) {}

    device_id_t device_id_{};
    std::shared_ptr<generic_allocation_handle> handle_;
};

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
    } -> std::same_as<const single_device_allocation_anchor&>;
};

template<class T, page_size_t PageSize = default_page_size>
class allocation_factory {
  public:
    using page_handle = page_handle<page_handle_type::weak, PageSize>;
    using page_vector = std::vector<page_handle>;

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
    auto allocate_pages(
        device_id_t device_id,
        const std::vector<page_index_t>& indices,
        const page_vector& weak_pages,
        Args&&... args
    ) -> single_device_allocation_anchor {
        using allocation = AllocationType<T, ReusePagesFlag, PageSize>;
        allocation
        alloc(device_id, indices, weak_pages, std::forward<Args>(args)...);
        add_pages_from_anchor(alloc.anchor());
        return alloc.anchor();
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
    auto allocate_pages(
        device_id_t device_id,
        page_index_t first,
        page_index_t last,
        const page_vector& weak_pages,
        Args&&... args
    ) -> single_device_allocation_anchor {
        using allocation = AllocationType<T, ReusePagesFlag, PageSize>;
        allocation
        alloc(device_id, first, last, weak_pages, std::forward<Args>(args)...);
        add_pages_from_anchor(alloc.anchor());
        return alloc.anchor();
    }

    void add_pages_from_anchor(const single_device_allocation_anchor& anchor) {
        const auto& handle = anchor.handle<PageSize>();
        auto& device_pages = pages_[anchor.device_id()];
        std::transform(
            handle.pages().begin(),
            handle.pages().end(),
            std::back_inserter(device_pages),
            [](auto& page) { return page_handle(page); }
        );
        anchors_.push_back(anchor);
    }

    auto pages(device_id_t device_id) -> page_vector& {
        return pages_[device_id];
    }

  private:
    std::unordered_map<device_id_t, page_vector> pages_;
    std::vector<single_device_allocation_anchor> anchors_;
};

}  // namespace sclx::detail
