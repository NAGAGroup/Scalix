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

template<pagination_type PaginationType, page_size_t PageSize>
class device_allocation : public allocation<PageSize> {};

template<page_size_t PageSize>
class device_allocation<pagination_type::contiguous, PageSize>
    : public allocation<PageSize> {
  public:
    device_allocation(
        device_id_t device_id,
        page_count_t page_count,
        page_size_t bytes_per_page
    ) {
        if (device_id != host_device_id) {
            auto device     = sclx::device::get_devices()[device_id - 1];
            sycl::queue q   = sycl::queue(device);
            auto alloc_type = sclx::usm::alloc::device;
            data_           = sclx::make_unique<std::byte[]>(
                q,
                alloc_type,
                page_count * bytes_per_page
            );
        } else {
            throw std::runtime_error(
                "Cannot allocate device_allocation for host"
            );
        }
    }

    [[nodiscard]] std::byte* data() const { return data_.get(); }

  private:
    sclx::unique_ptr<std::byte[]> data_;
};

template<page_size_t PageSize>
using continguous_device_allocation
    = device_allocation<pagination_type::contiguous, PageSize>;

template<page_size_t PageSize>
struct allocation_methods<continguous_device_allocation, PageSize> {

    template<class T>
    static std::vector<page_handle<page_handle_type::strong, PageSize>>
    allocate_pages_and_reuse_if_possible(device_id_t device_id, const std::vector<page_index_t>& indices, const std::vector<page_handle<page_handle_type::weak, PageSize>>&) {
        auto page_count     = static_cast<page_count_t>(indices.size());
        auto bytes_per_page = page_traits<T>::allocated_bytes_per_page;
        auto allocation_ptr
            = std::make_shared<continguous_device_allocation<PageSize>>(
                device_id,
                page_count,
                bytes_per_page
            );
        std::vector<page_handle<page_handle_type::strong, PageSize>> pages;
        for (auto p = 0; p < page_count; ++p) {
            auto page_index      = indices[p];
            auto page_ptr        = allocation_ptr->data() + p * bytes_per_page;
            auto device_page_ptr = std::make_shared<device_page<PageSize>>(
                page_ptr,
                device_id,
                page_index,
                bytes_per_page
            );
            pages.emplace_back(
                make_page_handle<page_handle_type::strong, PageSize>(
                    device_page_ptr,
                    allocation_ptr
                )
            );
        }

        return pages;
    }

    template<class T>
    static std::vector<page_handle<page_handle_type::strong, PageSize>>
    allocate_pages_and_reuse_if_possible(device_id_t device_id, page_index_t first, page_index_t last, const std::vector<page_handle<page_handle_type::weak, PageSize>>&) {
        page_count_t page_count = last - first;
        auto bytes_per_page     = page_traits<T>::allocated_bytes_per_page;
        auto allocation_ptr
            = std::make_shared<continguous_device_allocation<PageSize>>(
                device_id,
                page_count,
                bytes_per_page
            );
        std::vector<page_handle<page_handle_type::strong, PageSize>> pages;
        for (auto p = 0; p < page_count; ++p) {
            auto page_index      = first + p;
            auto page_ptr        = allocation_ptr->data() + p * bytes_per_page;
            auto device_page_ptr = std::make_shared<device_page<PageSize>>(
                page_ptr,
                device_id,
                page_index,
                bytes_per_page
            );
            pages.emplace_back(
                make_page_handle<page_handle_type::strong, PageSize>(
                    device_page_ptr,
                    allocation_ptr
                )
            );
        }

        return pages;
    }
};

}  // namespace sclx::detail
