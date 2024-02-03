// BSD 3-Clause License
//
// Copyright (c) 2023-2024 Jack Myers
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

#pragma once

#include "host_page.hpp"
#include "pagination_traits.hpp"

namespace sclx::detail {

template<pagination_type, class, reuse_pages, page_size_t>
class host_allocation;

template<class T, reuse_pages ReusePagesFlag, page_size_t PageSize>
class host_allocation<
    pagination_type::contiguous,
    T,
    ReusePagesFlag,
    PageSize> {
public:
    struct allocation_handle : sclx::detail::allocation_handle<PageSize> {
        std::unique_ptr<sclx::byte_array> data_;
        std::vector<page_handle<page_handle_type::strong, PageSize>> pages_;

        [[nodiscard]] auto device_id() const -> device_id_t final {
            return sclx::host_device_id;
        }

        [[nodiscard]] auto pages() const -> const std::vector<
            page_handle<page_handle_type::strong, PageSize>>& final {
            return pages_;
        }
    };

    host_allocation(
        device_id_t device_id,
        const std::vector<page_index_t>& indices,
        const std::vector<
            page_handle<page_handle_type::weak, PageSize>>& /*unused*/
    ) : anchor_(access_anchor_interface::create_anchor<allocation_handle>()) {
        if (device_id != sclx::host_device_id) {
            throw std::runtime_error(
                "host_allocation: device_id must be host_device_id"
            );
        }
        auto page_count     = static_cast<page_count_t>(indices.size());
        auto bytes_per_page = page_traits<T>::allocated_bytes_per_page;
        auto& alloc_handle  = static_cast<allocation_handle&>(
            access_anchor_interface::get_allocation<PageSize>(anchor_)
        );
        alloc_handle.data_ = std::make_unique<sclx::byte_array>(
            static_cast<size_t>(page_count) * bytes_per_page
        );
        auto& pages = alloc_handle.pages_;
        for (page_index_t pidx = 0; pidx < page_count; ++pidx) {
            auto page_index = indices[pidx];
            auto page_ptr   = alloc_handle.data_.get() + pidx * bytes_per_page;
            pages.emplace_back(
                make_page_handle<sclx::detail::host_page<PageSize>>(
                    page_ptr,
                    device_id,
                    page_index,
                    bytes_per_page
                )
            );
        }
    }

    host_allocation(
        device_id_t device_id,
        page_index_t first,
        page_index_t last,
        const std::vector<
            page_handle<page_handle_type::weak, PageSize>>& /*unused*/
    ) : anchor_(access_anchor_interface::create_anchor<allocation_handle>()) {
        auto page_count     = last - first;
        auto bytes_per_page = page_traits<T>::allocated_bytes_per_page;
        auto& alloc_handle  = static_cast<allocation_handle&>(
            access_anchor_interface::get_allocation<PageSize>(anchor_)
        );
        alloc_handle.data_ = std::make_unique<sclx::byte_array>(
            static_cast<size_t>(page_count) * bytes_per_page
        );
        auto& pages = alloc_handle.pages_;
        for (page_index_t pidx = 0; pidx < page_count; ++pidx) {
            auto page_index = first + pidx;
            auto page_ptr   = alloc_handle.data_.get() + pidx * bytes_per_page;
            pages.emplace_back(
                make_page_handle<sclx::detail::host_page<PageSize>>(
                    page_ptr,
                    device_id,
                    page_index,
                    bytes_per_page
                )
            );
        }
    }

    [[nodiscard]] auto anchor() const -> const access_anchor& {
        return this->anchor_;
    }

  private:
    access_anchor anchor_;
};

template<class T, reuse_pages ReusePagesFlag, page_size_t PageSize>
class host_allocation<pagination_type::paginated, T, ReusePagesFlag, PageSize> {
public:
    struct allocation_handle : sclx::detail::allocation_handle<PageSize> {
        std::vector<std::unique_ptr<sclx::byte_array>> raw_page_data_;
        std::vector<page_handle<page_handle_type::strong, PageSize>> pages_;

        [[nodiscard]] auto device_id() const -> device_id_t final {
            return sclx::host_device_id;
        }

        [[nodiscard]] auto pages() const -> const std::vector<
            page_handle<page_handle_type::strong, PageSize>>& final {
            return pages_;
        }
    };

    host_allocation(
        device_id_t device_id,
        const std::vector<page_index_t>& indices,
        const std::vector<page_handle<page_handle_type::weak, PageSize>>&
            existing_pages
    )
        : anchor_(access_anchor_interface::create_anchor<allocation_handle>()) {
        auto page_count     = static_cast<page_count_t>(indices.size());
        auto bytes_per_page = page_traits<T>::allocated_bytes_per_page;
        auto& alloc_handle  = static_cast<allocation_handle&>(
            access_anchor_interface::get_allocation<PageSize>(anchor_)
        );
        auto& pages = alloc_handle.pages_;
        for (page_index_t pidx = 0; pidx < page_count; ++pidx) {
            auto page_index           = indices[pidx];
            auto locked_existing_page = existing_pages[pidx].lock();
            if (existing_pages[page_index].lock().is_valid()) {
                pages.emplace_back(std::move(locked_existing_page));
                continue;
            }
            alloc_handle.raw_page_data_.emplace_back(
                std::make_unique<sclx::byte_array>(bytes_per_page)
            );
            auto page_ptr = alloc_handle.raw_page_data_.back().get();
            pages.emplace_back(
                make_page_handle<sclx::detail::host_page<PageSize>>(
                    page_ptr,
                    device_id,
                    page_index,
                    bytes_per_page
                )
            );
        }
    }

    host_allocation(
        device_id_t device_id,
        page_index_t first,
        page_index_t last,
        const std::vector<page_handle<page_handle_type::weak, PageSize>>&
            existing_pages
    )
        : anchor_(access_anchor_interface::create_anchor<allocation_handle>()) {
        auto page_count     = last - first;
        auto bytes_per_page = page_traits<T>::allocated_bytes_per_page;
        auto& alloc_handle  = static_cast<allocation_handle&>(
            access_anchor_interface::get_allocation<PageSize>(anchor_)
        );
        auto device = sclx::device::get_devices()[device_id];
        sycl::queue queue{device};
        auto& pages = alloc_handle.pages_;
        for (page_index_t pidx = 0; pidx < page_count; ++pidx) {
            auto page_index           = first + pidx;
            auto locked_existing_page = existing_pages[pidx].lock();
            if (existing_pages[page_index].lock().is_valid()) {
                pages.emplace_back(std::move(locked_existing_page));
                continue;
            }
            alloc_handle.raw_page_data_.emplace_back(
                std::make_unique<sclx::byte_array>(bytes_per_page)
            );
            auto page_ptr = alloc_handle.raw_page_data_.back().get();
            pages.emplace_back(
                make_page_handle<sclx::detail::host_page<PageSize>>(
                    page_ptr,
                    device_id,
                    page_index,
                    bytes_per_page
                )
            );
        }
    }

    [[nodiscard]] auto anchor() const -> const access_anchor& {
        return anchor_;
    }

  private:
    access_anchor anchor_;
};

template class host_allocation<
    pagination_type::contiguous,
    float,
    reuse_pages::if_possible,
    default_page_size>;

}  // namespace sclx::detail
