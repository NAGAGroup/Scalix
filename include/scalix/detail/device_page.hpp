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

#include "page_handle.hpp"

namespace sclx::detail {

template<page_size_t PageSize>
class device_page_table;

template<page_size_t PageSize>
class device_page : public page_interface<PageSize> {
  public:
    using page_interface = sclx::detail::page_interface<PageSize>;

    device_page() = default;

    device_page(
        sclx::byte* data,
        device_id_t device_id,
        page_index_t index,
        page_size_t allocated_bytes_per_page
    )
        : data_(data),
          device_id_(device_id),
          index_(index),
          allocated_bytes_per_page_(allocated_bytes_per_page) {}

    std::variant<sclx::byte*, std::future<sclx::byte*>> data() override {
        return data_;
    }

    std::variant<device_id_t, sclx::mpi_device> device_id() override {
        return device_id_;
    }

    std::variant<sclx::write_bit_t, std::future<write_bit_t>>
    write_bit() override {
        return write_bit_;
    }

    sclx::event set_write_bit(sclx::write_bit_t bit) override {
        write_bit_ = bit;
        return {};
    }

    page_index_t index() override { return index_; }

    sclx::event
    copy_from(sycl::queue q, std::shared_ptr<page_interface> src) override {
        auto src_device_id_variant = src->device_id();
        device_id_t src_device_id;
        if (src->is_mpi_local()) {
            src_device_id = std::get<device_id_t>(src_device_id_variant);
            if (src_device_id == no_device) {
                throw std::runtime_error(
                    "Source page has no device associated with it, cannot "
                    "copy from it. This error indicates a bug in Scalix, please "
                    "report it."
                );
            }
        }  else {
            throw std::runtime_error(
                "Device pages cannot copy from MPI devices yet."
            );
        }

        if (device_id_ == no_device) {
            throw std::runtime_error(
                "Device page has no device associated with it, cannot copy "
                "to it. This error indicates a bug in Scalix, please report "
                "it."
            );
        }

        auto src_data_variant = src->data();
        auto src_data = std::get<sclx::byte*>(src_data_variant);
        if (src_data == data_) {
            return {};
        }
        return q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(data_, src_data, allocated_bytes_per_page_);
        });
    }

    friend class device_page_table<PageSize>;

  private:
    sclx::byte* data_{nullptr};
    write_bit_t write_bit_{0};
    device_id_t device_id_{-1};
    page_index_t index_{0};
    page_size_t allocated_bytes_per_page_{0};
};

}  // namespace sclx::detail
