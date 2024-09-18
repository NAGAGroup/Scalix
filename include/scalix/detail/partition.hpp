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
#include <hipSYCL/sycl/info/event.hpp>
#include <hipSYCL/sycl/queue.hpp>
#include <memory>
#include <scalix/concurrent_guard.hpp>
#include <scalix/defines.hpp>
#include <scalix/pointers.hpp>
#include <sycl/sycl.hpp>
#include <utility>

namespace sclx::detail {

struct partition_interface {

    // protected:

  public:
    using write_bit_t = unsigned char;

    partition_interface() = default;

    partition_interface(const partition_interface&)                    = delete;
    auto operator=(const partition_interface&) -> partition_interface& = delete;
    partition_interface(partition_interface&&)                    = default;
    auto operator=(partition_interface&&) -> partition_interface& = default;

    virtual void allocate()                 = 0;
    virtual void allocate(usm::alloc alloc) = 0;

    virtual auto copy_to(std::shared_ptr<partition_interface> dst
    ) -> sycl::event = 0;

    virtual auto copy_from(partition_interface* src) -> sycl::event = 0;

    [[nodiscard]] virtual auto pointer() -> void* = 0;

    [[nodiscard]] virtual auto write_bits_pointer() -> write_bit_t* = 0;

    [[nodiscard]] virtual auto queue() const -> sycl::queue = 0;

    [[nodiscard]] virtual auto operator==(const partition_interface& other
    ) const -> bool = 0;

    [[nodiscard]] virtual auto operator!=(const partition_interface& other
    ) const -> bool = 0;

    virtual ~partition_interface() = default;
};

template<class T>
struct typed_partition : public partition_interface {
    struct copy_events_t {
        std::vector<sycl::event> read_events;
        std::vector<sycl::event> write_events;
    };

  public:
    using partition_interface::write_bit_t;

    typed_partition() = default;

    typed_partition(const typed_partition&)                    = delete;
    auto operator=(const typed_partition&) -> typed_partition& = delete;
    typed_partition(typed_partition&&)                         = default;
    auto operator=(typed_partition&&) -> typed_partition&      = default;

    void allocate() override { this->allocate(usm::alloc::shared); }

    void allocate(usm::alloc alloc) override {
        if (data_ != nullptr) {
            return;
        }
        data_ = sclx::make_shared<T[]>(assoc_queue_, alloc, part_elements_);
        write_bits_ = sclx::make_shared<write_bit_t[]>(
            assoc_queue_,
            alloc,
            part_elements_
        );
    }

    [[nodiscard]] auto queue() const -> sycl::queue override {
        return assoc_queue_;
    }

    auto copy_to(std::shared_ptr<partition_interface> dst
    ) -> sycl::event override {
        return this->copy_to_imp(std::static_pointer_cast<typed_partition>(dst)
        );
    }

    auto copy_from(partition_interface* src) -> sycl::event override {
        this->copy_from_imp(static_cast<typed_partition*>(src));
    }
    [[nodiscard]] auto pointer() -> void* override { return data_.get(); }

    [[nodiscard]] auto write_bits_pointer() -> write_bit_t* override {
        return write_bits_.get();
    }

    [[nodiscard]] auto operator==(const partition_interface& other
    ) const -> bool override {
        return *this == static_cast<const typed_partition&>(other);
    }

    [[nodiscard]] auto operator!=(const partition_interface& other
    ) const -> bool override {
        return !(*this == other);
    }

    [[nodiscard]] auto operator==(const typed_partition& other) const -> bool {
        return data_ == other.data_ && part_elements_ == other.part_elements_
            && write_bits_ == other.write_bits_;
    }

    [[nodiscard]] auto operator!=(const typed_partition& other) const -> bool {
        return !(*this == other);
    }

    ~typed_partition() override {
        auto copy_events_view
            = copy_events_.template get_view<access_mode::write>();
        for (sycl::event& event : copy_events_view->read_events) {
            event.wait_and_throw();
        }
        for (sycl::event& event : copy_events_view->write_events) {
            event.wait_and_throw();
        }
    }

    typed_partition(const sycl::queue& assoc_queue, size_t part_elements)
        : assoc_queue_{assoc_queue},
          part_elements_{part_elements} {}

    // private:
    auto copy_to_imp(std::shared_ptr<typed_partition> dst) -> sycl::event {
        return dst->copy_from_imp(this);
    }

    auto copy_from_imp(typed_partition* src) -> sycl::event {
        if (src->pointer() == this->pointer()) {
            return assoc_queue_.submit([](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<>{1}, [](sycl::id<>) {});
            });
        }
        sycl::event data_event;
        {
            auto this_events_view
                = copy_events_.template get_view<access_mode::write>();
            // std::erase_if(
            //     this_events_view->read_events,
            //     [](sycl::event& event) {
            //         return event.get_info<
            //                    sycl::info::event::command_execution_status>()
            //             == sycl::info::event_command_status::complete;
            //     }
            // );
            // std::erase_if(
            //     this_events_view->write_events,
            //     [](sycl::event& event) {
            //         return event.get_info<
            //                    sycl::info::event::command_execution_status>()
            //             == sycl::info::event_command_status::complete;
            //     }
            // );

            auto src_events_view
                = src->copy_events_.template get_view<access_mode::write>();
            // std::erase_if(src_events_view->read_events, [](sycl::event&
            // event) {
            //     return event.get_info<
            //                sycl::info::event::command_execution_status>()
            //         == sycl::info::event_command_status::complete;
            // });
            // std::erase_if(
            //     src_events_view->write_events,
            //     [](sycl::event& event) {
            //         return event.get_info<
            //                    sycl::info::event::command_execution_status>()
            //             == sycl::info::event_command_status::complete;
            //     }
            // );

            // when adding dependent events from the dest partition, we can
            // move/reset the read events because the write event created
            // from this call will depend on them
            std::vector<sycl::event> dep_events(this_events_view->read_events);
            std::transform(
                this_events_view->write_events.begin(),
                this_events_view->write_events.end(),
                std::back_inserter(dep_events),
                [](auto& event) { return event; }
            );
            this_events_view->write_events.clear();

            // for the source partition, we don't care about read events,
            // however we can still erase completed events to free up space
            std::transform(
                src_events_view->write_events.begin(),
                src_events_view->write_events.end(),
                dep_events.begin(),
                [](auto& event) { return event; }
            );

            auto data_event
                = assoc_queue_.submit([=, &dep_events, this](sycl::handler& cgh
                                      ) {
                      cgh.depends_on(dep_events);
                      cgh.memcpy(
                          src->pointer(),
                          data_.get(),
                          part_elements_ * sizeof(T)
                      );
                  });
            auto write_bits_event
                = assoc_queue_.submit([=, &dep_events, this](sycl::handler& cgh
                                      ) {
                      cgh.depends_on(dep_events);
                      cgh.memcpy(
                          src->write_bits_pointer(),
                          write_bits_.get(),
                          part_elements_ * sizeof(write_bit_t)
                      );
                  });

            this_events_view->write_events.push_back(data_event);
            this_events_view->write_events.push_back(write_bits_event);

            src_events_view->read_events.push_back(data_event);
            src_events_view->read_events.push_back(write_bits_event);

            data_event = assoc_queue_.submit([&](sycl::handler& cgh) {
                cgh.depends_on(data_event);
                cgh.depends_on(write_bits_event);
                cgh.parallel_for(sycl::range<>{1}, [](sycl::id<>) {});
            });
        }

        return data_event;
    }

    sycl::queue assoc_queue_;
    ::sclx::shared_ptr<T> data_;
    ::sclx::shared_ptr<write_bit_t> write_bits_;
    size_t part_elements_{};
    concurrent_guard<copy_events_t> copy_events_;
};

template<class T, int Dimensions>
struct nd_partition : public typed_partition<T> {

    // protected:
    using base = typed_partition<T>;

  public:
    static auto create_empty_partition(
        const sycl::queue& assoc_queue,
        sycl::range<Dimensions> shape
    ) -> std::shared_ptr<nd_partition> {
        return std::make_shared<nd_partition>(assoc_queue, shape);
    }

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
            / bytes_per_leading_index;
        auto elements_per_part
            = leading_elements_per_part * elements_per_leading_index;
        std::vector<std::shared_ptr<nd_partition>> part_list;
        for (size_t i = 0; i < combined_shape[0]; i += elements_per_part) {
            size_t leading_idx_end
                = (i + leading_elements_per_part > combined_shape[0])
                    ? combined_shape[0]
                    : i + leading_elements_per_part;
            auto leading_elements = leading_idx_end - i;
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

    nd_partition(const sycl::queue& assoc_queue, sycl::range<Dimensions> shape)
        : base::typed_partition(assoc_queue, shape.size()),
          part_shape_{shape} {}

    // private:
    sycl::range<Dimensions> part_shape_{};
};

}  // namespace sclx::detail
