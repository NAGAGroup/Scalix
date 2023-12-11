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

#include "page_table.hpp"
#include <scalix/find_device.hpp>

namespace sclx::detail {

template<page_size_t PageSize = default_page_size>
class device_page_table : public page_table_interface<PageSize> {
  public:
    using interface   = sclx::detail::page_table_interface<PageSize>;
    using page_handle = typename interface::page_handle;
    using iterator    = typename interface::iterator;

    device_page_table(const device& device, page_count_t page_count)
        : q_(device),
          device_id_(find_device(device)),
          pages_(page_count),
          host_staging_page_data_(page_count),
          device_accessible_page_data_(sclx::make_unique<page_data<PageSize>[]>(
              q_,
              sclx::usm::alloc::device,
              page_count
          )) {}

    auto map_page(page_handle page) -> sclx::event override {
        auto page_lock = page.lock();
        if (page_lock.index() == sclx::invalid_page) {
            throw std::runtime_error(
                "Cannot map an invalid page. This error indicates a bug in "
                "Scalix, please report it."
            );
        }
        return pages_[page_lock.index()].lock().replace_with(q_, page_lock);
    }

    auto device_id() -> std::variant<device_id_t, sclx::mpi_device> override {
        return device_id_;
    }

    auto unmap_invalid_pages() -> sclx::event override {
        for (auto& page : pages_) {
            if (page.lock().is_valid()) {
                page.release();
            }
        }

        return {};
    }

    auto begin() -> iterator override { return pages_.begin(); }

    auto end() -> iterator override { return pages_.end(); }

    auto make_table_host_accessible() -> sclx::event override {
        auto copy_queue = q_.memcpy(
            host_staging_page_data_.data(),
            device_accessible_page_data_.get(),
            pages_.size() * sizeof(page_data<PageSize>)
        );
        return q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(copy_queue);
            cgh.host_task([&]() {
                for (std::size_t i = 0; i < pages_.size(); ++i) {
                    pages_[i].lock().set_write_bit(
                        host_staging_page_data_[i].write_bit
                    );
                }
            });
        });
    }

    auto make_table_device_accessible() -> sclx::event override {
        auto staging_event = q_.submit([&](sycl::handler& cgh) {
            cgh.host_task([=, this]() {
                for (std::size_t i = 0; i < pages_.size(); ++i) {
                    auto page_lock = pages_[i].lock();
                    if (!page_lock.is_mpi_local()) {
                        throw std::runtime_error(
                            "Device page handles should not exist on other MPI "
                            "nodes. "
                            "Assuming "
                            "bad page. This error indicates a bug in Scalix, "
                            "please "
                            "report it."
                        );
                    }

                    auto data_variant = page_lock.data();
                    auto data         = std::get<sclx::byte*>(data_variant);
                    host_staging_page_data_[i].data = data;
                    auto write_bit_variant          = page_lock.write_bit();
                    auto write_bit
                        = std::get<sclx::write_bit_t>(write_bit_variant);
                    host_staging_page_data_[i].write_bit = write_bit;
                }
            });
        });

        return q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(staging_event);
            cgh.memcpy(
                device_accessible_page_data_.get(),
                host_staging_page_data_.data(),
                pages_.size() * sizeof(page_data<PageSize>)
            );
        });
    }

  private:
    sycl::queue q_;
    device_id_t device_id_;
    std::vector<page_handle> pages_;
    std::vector<page_data<PageSize>> host_staging_page_data_;
    sclx::unique_ptr<page_data<PageSize>[]> device_accessible_page_data_;
};

}  // namespace sclx::detail