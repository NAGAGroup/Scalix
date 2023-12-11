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

#include "allocation.hpp"
#include "device_page.hpp"

namespace sclx::detail {

enum class pagination_type { contiguous, paginated };

template<
    pagination_type PaginationType,
    class T,
    reuse_pages ReusePagesFlag,
    page_size_t PageSize>
class device_allocation;

template<pagination_type PaginationType>
struct device_pagination_traits {
    template<class T, reuse_pages ReusePagesFlag, page_size_t PageSize>
    using allocation_type
        = device_allocation<PaginationType, T, ReusePagesFlag, PageSize>;
};

template<class T, reuse_pages ReusePagesFlag, page_size_t PageSize>
class device_allocation<
    pagination_type::contiguous,
    T,
    ReusePagesFlag,
    PageSize> {
  public:
    device_allocation(
        device_id_t device_id,
        const std::vector<page_index_t>& indices,
        const std::vector<
            page_handle<page_handle_type::weak, PageSize>>& /*unused*/
    ) {
        auto page_count     = static_cast<page_count_t>(indices.size());
        auto bytes_per_page = page_traits<T>::allocated_bytes_per_page;
        allocation_handle alloc_handle;
        auto device = sclx::device::get_devices()[device_id];
        sycl::queue queue{device};
        alloc_handle.data_ = sclx::make_unique<sclx::byte_array>(
            queue,
            sclx::usm::alloc::device,
            page_count * bytes_per_page
        );
        auto& pages = alloc_handle.pages_;
        for (page_index_t pidx = 0; pidx < page_count; ++pidx) {
            auto page_index = indices[pidx];
            auto page_ptr   = alloc_handle.data_.get() + pidx * bytes_per_page;
            auto device_page_ptr = std::make_shared<device_page<PageSize>>(
                page_ptr,
                device_id,
                page_index,
                bytes_per_page
            );
            pages.emplace_back(
                make_page_handle<page_handle_type::strong, PageSize>(
                    device_page_ptr
                )
            );
        }
        anchor_ = make_access_anchor(device_id, std::move(alloc_handle));
    }

    device_allocation(
        device_id_t device_id,
        page_index_t first,
        page_index_t last,
        const std::vector<
            page_handle<page_handle_type::weak, PageSize>>& /*unused*/
    ) {
        auto page_count     = last - first;
        auto bytes_per_page = page_traits<T>::allocated_bytes_per_page;
        allocation_handle alloc_handle;
        auto device = sclx::device::get_devices()[device_id];
        sycl::queue queue{device};
        alloc_handle.data_ = sclx::make_unique<sclx::byte_array>(
            queue,
            sclx::usm::alloc::device,
            page_count * bytes_per_page
        );
        auto& pages = alloc_handle.pages_;
        for (page_index_t pidx = 0; pidx < page_count; ++pidx) {
            auto page_index = first + pidx;
            auto page_ptr   = alloc_handle.data_.get() + pidx * bytes_per_page;
            auto device_page_ptr = std::make_shared<device_page<PageSize>>(
                page_ptr,
                device_id,
                page_index,
                bytes_per_page
            );
            pages.emplace_back(
                make_page_handle<page_handle_type::strong, PageSize>(
                    device_page_ptr
                )
            );
        }
        anchor_ = make_access_anchor(device_id, std::move(alloc_handle));
    }

    [[nodiscard]] auto anchor() const
        -> const single_device_allocation_anchor& {
        return this->anchor_;
    }

    struct allocation_handle : sclx::detail::allocation_handle<PageSize> {
        using page_handle
            = sclx::detail::page_handle<page_handle_type::strong, PageSize>;
        sclx::unique_ptr<sclx::byte_array> data_;
        std::vector<page_handle> pages_;

        auto pages() const -> const std::vector<page_handle>& override {
            return pages_;
        }
    };

  private:
    single_device_allocation_anchor anchor_;
};

}  // namespace sclx::detail
