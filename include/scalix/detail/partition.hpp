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
#include <hipSYCL/sycl/event.hpp>
#include <hipSYCL/sycl/handler.hpp>
#include <hipSYCL/sycl/queue.hpp>
#include <memory>
#include <scalix/concurrent_guard.hpp>
#include <scalix/defines.hpp>
#include <scalix/pointers.hpp>
#include <sycl/sycl.hpp>
#include <utility>

namespace sclx::detail {

class partition_interface {

  protected:

  public:
    partition_interface() = default;

    partition_interface(const partition_interface&)                    = delete;
    auto operator=(const partition_interface&) -> partition_interface& = delete;
    partition_interface(partition_interface&&)                    = default;
    auto operator=(partition_interface&&) -> partition_interface& = default;

    virtual void allocate(usm::alloc alloc) = 0;

    virtual auto copy_to(std::shared_ptr<partition_interface> dst
    ) -> sycl::event = 0;

    virtual auto copy_from(partition_interface* src) -> sycl::event = 0;

    [[nodiscard]] virtual auto pointer() -> void* = 0;

    [[nodiscard]] virtual auto operator==(const partition_interface& other
    ) const -> bool = 0;

    [[nodiscard]] virtual auto operator!=(const partition_interface& other
    ) const -> bool = 0;

    virtual ~partition_interface() = default;
};

template<class T>
class typed_partition : partition_interface {
    struct copy_events_t {
        std::vector<sycl::event> read_events;
        std::unique_ptr<sycl::event> latest_write_event{nullptr};
    };

  public:
    typed_partition() = default;

    typed_partition(const typed_partition&)                    = delete;
    auto operator=(const typed_partition&) -> typed_partition& = delete;
    typed_partition(typed_partition&&)                         = default;
    auto operator=(typed_partition&&) -> typed_partition&      = default;

    void allocate(usm::alloc alloc) override {
        if (part_ptr_ != nullptr) {
            return;
        }
        part_ptr_ = sclx::make_unique<T[]>(assoc_queue_, alloc, part_elements_);
    }

    auto copy_to(std::shared_ptr<typed_partition> dst) -> sycl::event override {
        return dst->copy_from(this);
    }
    auto copy_from(typed_partition* src) -> sycl::event override {
        if (src->pointer() == this->pointer()) {
            return assoc_queue_.submit([](sycl::handler& cgh) {
                cgh.single_task([] {});
            });
        }
        sycl::event event;
        {
            auto this_events_view
                = copy_events_.template get_view<access_mode::write>();
            auto src_events_view
                = src->copy_events_.template get_view<access_mode::write>();

            // when adding dependent events from the dest partition, we can
            // move/reset the read events because the write event created from
            // this call will depend on them
            std::vector<sycl::event> dep_events(
                std::move(this_events_view->read_events)
            );
            if (this_events_view->write_event != nullptr) {
                dep_events.push_back(*this_events_view->write_event);
            } else {
                this_events_view->write_event = std::unique_ptr<sycl::event>();
            }

            // for the source partition, we don't care about read events
            if (src_events_view->write_event != nullptr) {
                dep_events.push_back(*src_events_view->write_event);
            }

            event
                = assoc_queue_.submit([=, &dep_events, this](sycl::handler& cgh
                                      ) {
                      cgh.depends_on(dep_events);
                      cgh.copy(
                          static_cast<T*>(src->pointer()),
                          part_ptr_,
                          part_elements_
                      );
                  });

            *this_events_view->write_event = event;

            src_events_view->read_events.push_back(event);
        }

        return event;
    }

    [[nodiscard]] auto pointer() -> void* override { return part_ptr_; }

    [[nodiscard]] auto operator==(const typed_partition& other
    ) const -> bool override {
        return part_ptr_ == other.part_ptr_
            && part_elements_ == other.part_elements_;
    }

    [[nodiscard]] auto operator!=(const typed_partition& other) const -> bool {
        return !(*this == other);
    }

    ~typed_partition() override {
        auto copy_events_view
            = copy_events_.template get_view<access_mode::read>();
        for (sycl::event& event : copy_events_view->read_events) {
            event.wait();
        }
        copy_events_view->write_event->wait();
    }

  protected:
    typed_partition(
        const sycl::queue& assoc_queue,
        T* part_ptr,
        size_t part_elements
    )
        : assoc_queue_{assoc_queue},
          part_ptr_{part_ptr},
          part_elements_{part_elements} {}

  private:
    sycl::queue assoc_queue_;
    std::unique_ptr<T*> part_ptr_;
    size_t part_elements_{};
    concurrent_guard<copy_events_t> copy_events_;
};

template<class T, int Dimensions>
class nd_partition : typed_partition<T> {

  protected:
    using base = typed_partition<T>;

  public:
    static auto create_empty_partitions(
        const sycl::queue& assoc_queue,
        sycl::range<Dimensions> combined_shape,
        size_t min_bytes_per_part
    ) -> std::vector<std::shared_ptr<nd_partition>> {
        size_t elements_per_leading_index = 1;
        for (int i = 1; i < Dimensions; ++i) {
            elements_per_leading_index *= combined_shape[i];
        }
        auto bytes_per_leading_index = elements_per_leading_index * sizeof(T);
        auto leading_elements_per_part
            = (bytes_per_leading_index + min_bytes_per_part - 1)
            / min_bytes_per_part;
        auto elements_per_part
            = leading_elements_per_part * elements_per_leading_index;
        std::vector<std::shared_ptr<nd_partition>> part_list;
        for (size_t i = 0; i < combined_shape[0]; i += elements_per_part) {
            size_t leading_idx_end = (i + elements_per_part > combined_shape)
                                       ? combined_shape[0]
                                       : i + elements_per_part;
            auto leading_elements  = leading_idx_end - i;
            auto num_elements = leading_elements * elements_per_leading_index;

            auto part_shape = combined_shape;
            part_shape[0]   = leading_elements;
            part_list.push_back(
                std::make_shared<nd_partition>(assoc_queue, part_shape)
            );
        }
        return part_list;
    }

    [[nodiscard]] auto shape() const -> const sycl::range<Dimensions>& {
        return part_shape_;
    }

  protected:
    nd_partition(const sycl::queue& assoc_queue, sycl::range<Dimensions> shape)
        : base::typed_partition(assoc_queue, shape.size()),
          part_shape_{shape} {}

  private:
    sycl::range<Dimensions> part_shape_{};
};

}  // namespace sclx::detail
