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

namespace sclx::detail {

template<page_size_t PageSize = default_page_size>
class host_page_table : public page_table_interface<PageSize> {
  public:
    using interface          = sclx::detail::page_table_interface<PageSize>;
    using weak_page_handle   = typename interface::weak_page_handle;
    using strong_page_handle = typename interface::strong_page_handle;
    using iterator           = typename interface::iterator;

    explicit host_page_table(page_count_t page_count)
        : pages_(page_count),
          simplified_page_data_(page_count) {}

    auto map_page(strong_page_handle page) -> sclx::event override {
        if (page.index() == sclx::invalid_page) {
            throw std::runtime_error(
                "Cannot map an invalid page. This error indicates a bug in "
                "Scalix, please report it."
            );
        }
        return pages_[page.index()].lock().replace_with(queue_, page);
    }

    auto device_id() -> std::variant<device_id_t, sclx::mpi_device> override {
        return sclx::host_device_id;
    }

    auto unmap_invalid_pages() -> sclx::event override {
        for (auto& page : pages_) {
            if (!page.lock().is_valid()) {
                page.release();
            }
        }

        return {};
    }

    auto begin() -> iterator override { return pages_.begin(); }

    auto end() -> iterator override { return pages_.end(); }

    auto pages() const -> const std::vector<weak_page_handle>& override {
        return pages_;
    }

    auto make_table_host_accessible() -> sclx::event override {
        return queue_.submit([&](sycl::handler& cgh) {
            cgh.host_task([&]() {
                for (std::size_t i = 0; i < pages_.size(); ++i) {
                    pages_[i].lock().set_write_bit(
                        simplified_page_data_[i].write_bit
                    );
                }
            });
        });
    }

    auto make_table_device_accessible() -> sclx::event override {
        return queue_.submit([&](sycl::handler& cgh) {
            cgh.host_task([&]() {
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
                    simplified_page_data_[i].write_bit
                        = std::get<sclx::write_bit_t>(page_lock.write_bit());
                    simplified_page_data_[i].data
                        = std::get<sclx::byte*>(page_lock.data());
                }
            });
        });
    }

  private:
    sycl::queue queue_{};
    std::vector<weak_page_handle> pages_{};
    std::vector<page_data<PageSize>> simplified_page_data_{};
};

}  // namespace sclx::detail
