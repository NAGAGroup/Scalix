// BSD 3-Clause License
//
// Copyright (c) 2024 Jack Myers
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
#include <algorithm>
#include <future>
#include <memory>
#include <scalix/concurrent_guard.hpp>
#include <scalix/defines.hpp>
#include <scalix/pointers.hpp>
#include <sycl/sycl.hpp>
#include <utility>

namespace sclx::detail {

class partition_interface {
  public:
    partition_interface() = default;

    partition_interface(const partition_interface&) = default;
    partition_interface(partition_interface&&)      = default;

    static sycl::event copy(
        sycl::queue source_queue,
        page_ptr_t source,
        sycl::queue dest_queue,
        page_ptr_t destination,
        page_size_t page_size
    ) {
        if (source == nullptr || destination == nullptr) {
            return {};
        }
        if (source == destination) {
            return {};
        }

        auto source_type
            = sycl::get_pointer_type(source, source_queue.get_context());
        auto dest_type
            = sycl::get_pointer_type(destination, dest_queue.get_context());

        if (source_type == usm::alloc::unknown
            || dest_type == usm::alloc::unknown) {
            throw std::invalid_argument{"Unknown USM type"};
        }

        if (source_queue.get_device() == dest_queue.get_device()) {
            auto event = dest_queue.memcpy(destination, source, page_size);
            //            event.wait_and_throw();
            return event;
        }

        page_ptr_t host_ptr = nullptr;
        ::sclx::unique_ptr<byte[]> host_ptr_owner;
        sycl::event host_copy_event;
        if (dest_type == usm::alloc::host) {
            host_ptr = source;
        } else if (source_type == usm::alloc::host) {
            host_ptr = destination;
        } else {
            host_ptr_owner = ::sclx::make_unique<byte[]>(
                source_queue,
                usm::alloc::host,
                page_size
            );
            host_ptr        = host_ptr_owner.get();
            host_copy_event = source_queue.memcpy(host_ptr, source, page_size);
        }

        auto event
            = dest_queue
                  .memcpy(destination, host_ptr, page_size, host_copy_event);
        //        event.wait_and_throw();
        return event;
    }

    auto
    operator=(const partition_interface&) -> partition_interface& = default;
    auto operator=(partition_interface&&) -> partition_interface& = default;

    virtual auto copy_to(partition_interface& other) const -> sycl::event = 0;
    virtual auto
    copy_to(sycl::queue dest_queue, page_ptr_t destination) const -> sycl::event
                                                                     = 0;

    [[nodiscard]] virtual auto
    copy_to(std::shared_ptr<partition_interface> other) const -> sycl::event
                                                                 = 0;

    virtual auto
    copy_from(sycl::queue source_queue, concurrent_guard<page_ptr_t> source)
        -> sycl::event = 0;

    virtual auto device_queue() const -> sycl::queue = 0;

    [[nodiscard]] virtual auto page_address() const -> const byte* = 0;

    [[nodiscard]] virtual auto check_if_same_page(page_ptr_t other
    ) const -> bool = 0;

    [[nodiscard]] virtual auto operator==(const partition_interface& other
    ) const -> bool = 0;

    [[nodiscard]] virtual auto operator!=(const partition_interface& other
    ) const -> bool = 0;

    virtual ~partition_interface() = default;
};

template<page_size_t PageSize>
class page_data final : public partition_interface {
  public:
    using alloc_handle_t            = std::shared_ptr<void>;
    static constexpr auto page_size = PageSize;

    page_data() = default;

    // ReSharper disable once CppParameterMayBeConst
    page_data(page_ptr_t data, alloc_handle_t alloc_handle, sycl::queue queue)
        : data_{data},
          alloc_handle_{std::move(alloc_handle)},
          queue_{std::move(queue)} {}

    auto copy_to(partition_interface& other) const -> sycl::event override {
        return other.copy_from(device_queue(), data_);
    }

    auto copy_to(sycl::queue dest_queue, page_ptr_t destination) const
        -> sycl::event override {
        auto data = data_.get_view<access_mode::read>();
        return partition_interface::copy(
            queue_,
            data.access(),
            dest_queue,
            destination,
            page_size
        );
    }

    auto copy_to(std::shared_ptr<partition_interface> other
    ) const -> sycl::event override {
        return copy_to(*other);
    }

    auto copy_from(
        sycl::queue source_queue,
        concurrent_guard<page_ptr_t> source_guard
    ) -> sycl::event override {
        if (source_guard.unsafe_access() == data_.unsafe_access()) {
            return {};
        }
        auto source = source_guard.get_view<access_mode::read>();
        auto dest   = data_.get_view<access_mode::write>();
        return partition_interface::copy(
            source_queue,
            source.access(),
            queue_,
            dest.access(),
            page_size
        );
    }

    auto device_queue() const -> sycl::queue override { return queue_; }

    [[nodiscard]] auto page_address() const -> const byte* override {
        return data_.unsafe_access();
    }

    // ReSharper disable once CppParameterMayBeConst
    [[nodiscard]] auto check_if_same_page(page_ptr_t other
    ) const -> bool override {
        if (!data_.valid() || other == nullptr) {
            return false;
        }
        return data_.unsafe_access() == other;
    }

    [[nodiscard]] auto operator==(const partition_interface& other
    ) const -> bool override {
        return other.check_if_same_page(data_.unsafe_access());
    }

    [[nodiscard]] auto operator!=(const partition_interface& other
    ) const -> bool override {
        return !(*this == other);
    }

  private:
    concurrent_guard<page_ptr_t> data_{nullptr};
    alloc_handle_t alloc_handle_;
    sycl::queue queue_;
};

}  // namespace sclx::detail
