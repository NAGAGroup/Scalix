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

#include "detail/partition.hpp"
#include "typed_task.hpp"
#include <hipSYCL/sycl/access.hpp>
#include <hipSYCL/sycl/queue.hpp>
#include <memory>
#include <scalix/access_anchor.hpp>
#include <scalix/concurrent_guard.hpp>
#include <scalix/generic_task.hpp>

namespace sclx {

template<class T, int Dimensions>
struct buffer;

struct access_strategy;

namespace detail {

template<class T, int Dimensions>
struct buffer_imp;

}

template<int Dimensions, int Iter = 0>
size_t linearize_id(
    const id<Dimensions>& index,
    const range<Dimensions>& range,
    size_t multiplier = 1
) {
    if constexpr (Iter == Dimensions) {
        return 0;
    } else {
        return linearize_id<Dimensions, Iter + 1>(
                   index,
                   range,
                   multiplier * range[Dimensions - Iter - 1]
               )
             + index[Dimensions - Iter - 1] * multiplier;
    }
}

template<class T, int Dimensions, access_mode AccessMode>
struct accessor {
    using partition_data_ptr      = T*;
    using partition_write_bit_ptr = unsigned char*;

    friend class detail::buffer_imp<T, Dimensions>;

  public:
    using value_type
        = std::conditional_t<AccessMode == access_mode::read, const T&, T&>;

    // template<
    //     class ReturnType
    //     = std::enable_if_t<AccessMode != access_mode::read, const
    //     value_type&>>
    // auto read_value(sycl::id<Dimensions> idx) const -> ReturnType {
    //     auto linear_idx = linearize_id(idx, shape_);
    //
    //     auto partition_idx = linear_idx / elements_per_partition_;
    //     auto element_idx   = linear_idx % elements_per_partition_;
    //     return (*partition_data_[partition_idx])[element_idx];
    // }
    //
    // template<
    //     class ReturnType
    //     = std::enable_if_t<AccessMode != access_mode::read, value_type&>>
    // auto
    // write_value(sycl::id<Dimensions> idx, const T& value) const -> ReturnType
    // {
    //     (*this)[idx] = value;
    //     return this->read_value(idx);
    // }

    auto operator[](sycl::id<Dimensions> idx) const -> value_type {
        auto linear_idx = linearize_id(idx, shape_);
        if constexpr (AccessMode != access_mode::read) {
            set_write_bit(linear_idx);
        }
        auto partition_idx = linear_idx / elements_per_partition_;
        auto element_idx   = linear_idx % elements_per_partition_;
        return (partition_data_[partition_idx])[element_idx];
    }

    // private:
    accessor(
        partition_data_ptr* partition_data,
        partition_write_bit_ptr* partition_write_bits,
        sycl::range<Dimensions> shape,
        size_t elements_per_partition
    )
        : partition_data_{partition_data},
          partition_write_bits_{partition_write_bits},
          shape_{shape},
          elements_per_partition_{elements_per_partition} {}

    void set_write_bit(size_t linear_idx) const {
        auto partition_idx = linear_idx / elements_per_partition_;
        auto element_idx   = linear_idx % elements_per_partition_;
        partition_write_bits_[partition_idx][element_idx] = 1;
    }

    partition_data_ptr* partition_data_;
    partition_write_bit_ptr* partition_write_bits_;
    sycl::range<Dimensions> shape_;
    size_t elements_per_partition_;
};

inline auto
future_to_event(sycl::queue queue, const std::shared_future<void>& fut)
    -> sycl::event {
    sycl::buffer<int, 1> buffer{1};
    std::promise<void> buffer_capture_promise;
    auto buffer_capture_future = buffer_capture_promise.get_future();
    std::thread thrd{[buffer_capture_promise
                      = std::move(buffer_capture_promise),
                      buffer,
                      fut]() mutable {
        auto acsr = buffer.get_access<sycl::access_mode::discard_write>();
        buffer_capture_promise.set_value();
        acsr[0] = 1;
        fut.get();
    }};
    thrd.detach();

    buffer_capture_future.get();
    sycl::buffer<int, 1> buffer_copy{1};
    auto event = queue.submit([=](sycl::handler& cgh) mutable {
        auto buffer_acc = buffer.get_access<sycl::access_mode::read>(cgh);
        auto buffer_copy_acc
            = buffer_copy.get_access<sycl::access_mode::write>(cgh);
        cgh.single_task([=]() { buffer_copy_acc[0] = buffer_acc[0]; });
    });

    return event;
}

struct queue {
    friend struct command_handler;

    struct global_metadata {
        std::vector<uint> sub_command_weights;
        std::vector<sycl::queue> sub_queues;
        std::vector<std::shared_ptr<void>> command_data;
        generic_task fetch_wait_task;
        generic_task command_wait_task;
        generic_task push_wait_task;
        std::atomic<int> sub_command_setup_completions;

        void notify_sub_command_setup_complete() {
            auto old = sub_command_setup_completions.fetch_add(1);
            if (static_cast<uint>(old + 1) == sub_queues.size()) {
                fetch_wait_task.launch();
                command_wait_task.launch();
                push_wait_task.launch();
            }
        }
    };

  public:
    template<class FunctionType>
    auto submit(FunctionType submission);

    queue(
        const std::vector<uint> sub_command_weights,
        const std::vector<sycl::queue> sub_queues
    )
        : sub_command_weights(sub_command_weights),
          sub_queues(sub_queues) {}

    // private:
    std::vector<uint> sub_command_weights;
    std::vector<sycl::queue> sub_queues;
};

struct command_handler {

    template<class T, int Dimensions>
    friend struct detail::buffer_imp;
    friend struct access_strategy;
    friend struct queue;

    enum class command_type { parallel_for };

    struct command_config {
        command_type cmd_type;
    };

  public:
    template<class CommandBody>
    void
    parallel_for(const range<1>& num_work_items, CommandBody command_body) {
        auto cmd_config_ptr
            = std::make_unique<command_config>(command_type::parallel_for);
        cmd_config_promise.set_value(std::move(cmd_config_ptr));
        global_metadata->notify_sub_command_setup_complete();

        const auto& weights = global_metadata->sub_command_weights;
        auto sum_weights    = std::accumulate(
            weights.begin(),
            weights.end(),
            0,
            std::plus<int>()
        );
        range<1> sub_command_start{};
        for (uint subcmd_idx = 0; subcmd_idx < sub_command_idx; ++subcmd_idx) {
            sub_command_start[0] += static_cast<uint>(
                static_cast<double>(weights[subcmd_idx]) / sum_weights
                * static_cast<double>(num_work_items[0])
            );
        }
        range<1> sub_command_end = num_work_items;
        if (sub_command_idx != global_metadata->sub_queues.size() - 1) {
            sub_command_end[0]
                = sub_command_start[0]
                + static_cast<uint>(
                      static_cast<double>(weights[sub_command_idx])
                      / sum_weights * static_cast<double>(num_work_items[0])
                );
        }

        range<1> sub_command_range = sub_command_end - sub_command_start;
        sycl_handler->parallel_for(sub_command_range, [=](sycl::id<> idx) {
            idx[0] += sub_command_start[0];
            command_body(idx);
        });
    }

    // private:
    command_handler(
        queue::global_metadata* global_metadata,
        uint sub_command_idx,
        sycl::handler* sycl_handler
    )
        : global_metadata(global_metadata),
          sub_command_idx(sub_command_idx),
          sycl_handler(sycl_handler) {}

    [[nodiscard]] auto sycl_queue() -> sycl::queue& {
        return global_metadata->sub_queues[sub_command_idx];
    }

    [[nodiscard]] auto
    cmd_config() -> std::future<std::unique_ptr<command_config>> {
        return cmd_config_promise.get_future();
    }

    [[nodiscard]] auto fetch_wait_task() -> generic_task& {
        return global_metadata->fetch_wait_task;
    }

    [[nodiscard]] auto command_wait_task() -> generic_task& {
        return global_metadata->command_wait_task;
    }

    [[nodiscard]] auto push_wait_task() -> generic_task& {
        return global_metadata->push_wait_task;
    }

    template<class T>
    void register_command_data(std::shared_ptr<T> ptr) {
        global_metadata->command_data.push_back(std::move(ptr));
    }

    template<class T>
    void register_command_data(std::vector<std::shared_ptr<T>> ptrs) {
        global_metadata->command_data.insert(
            global_metadata->command_data.end(),
            std::make_move_iterator(ptrs.begin()),
            std::make_move_iterator(ptrs.end())
        );
    }

    queue::global_metadata* global_metadata;
    uint sub_command_idx;
    sycl::handler* sycl_handler;
    std::promise<std::unique_ptr<command_handler::command_config>>
        cmd_config_promise;
};

struct access_strategy {
    template<class T, int Dimensions>
    friend struct ::sclx::detail::buffer_imp;

  public:
    virtual ~access_strategy() = default;

    // protected:
    using command_config = command_handler::command_config;
    access_strategy()    = default;

    access_strategy(const access_strategy&)                    = default;
    auto operator=(const access_strategy&) -> access_strategy& = default;
    access_strategy(access_strategy&&)                         = default;
    auto operator=(access_strategy&&) -> access_strategy&      = default;

    [[nodiscard]] virtual auto execute_strategy(
        std::vector<std::shared_ptr<detail::partition_interface>>& partitions,
        std::future<std::unique_ptr<command_handler::command_config>> cmd_config
    ) -> generic_task = 0;
};

namespace detail {

struct buffer_handle {
  public:
    buffer_handle()                                        = default;
    buffer_handle(const buffer_handle&)                    = default;
    auto operator=(const buffer_handle&) -> buffer_handle& = default;
    buffer_handle(buffer_handle&&)                         = default;
    auto operator=(buffer_handle&&) -> buffer_handle&      = default;

    virtual ~buffer_handle() = default;
};

template<class T, int Dimensions>
struct buffer_interface : public buffer_handle {
    // protected:
    [[nodiscard]] virtual auto get_read_access(
        command_handler& cgh,
        std::unique_ptr<access_strategy> strategy
    ) const -> accessor<T, Dimensions, access_mode::read> = 0;

    [[nodiscard]] virtual auto get_write_access(
        command_handler& cgh,
        std::unique_ptr<access_strategy> strategy
    ) const -> accessor<T, Dimensions, access_mode::write> = 0;
};

template<class T, int Dimensions>
struct buffer_imp : public buffer_interface<T, Dimensions> {
    static constexpr partition_size_t partition_size = 4096;
    using raw_partition_data_ptr                     = T*;
    using raw_partition_write_bits_ptr               = unsigned char*;
    using partition_ptr      = std::shared_ptr<nd_partition<T, Dimensions>>;
    using weak_partition_ptr = std::weak_ptr<nd_partition<T, Dimensions>>;

    friend struct ::sclx::buffer<T, Dimensions>;

    struct queue_commands {
        std::vector<generic_task> read_queue_commands_;
        std::vector<generic_task> write_queue_commands_;
    };

    struct partition_metadata {
        std::vector<partition_ptr> primary_partitions_;
        std::vector<std::vector<weak_partition_ptr>>
            potentially_active_partitions_;
    };

    explicit buffer_imp(sycl::range<Dimensions> shape) : shape_(shape) {}

    [[nodiscard]] auto get_read_access(
        command_handler& cgh,
        std::unique_ptr<access_strategy> strategy
    ) const -> accessor<T, Dimensions, access_mode::read> override {
        return this->get_access<access_mode::read>(cgh, strategy);
    }

    [[nodiscard]] auto get_write_access(
        command_handler& cgh,
        std::unique_ptr<access_strategy> strategy
    ) const -> accessor<T, Dimensions, access_mode::write> override {
        return this->get_access<access_mode::write>(cgh, strategy);
    }

    template<access_mode AccessMode>
    [[nodiscard]] auto
    get_access(command_handler& cgh, std::unique_ptr<access_strategy>& strategy)
        const -> accessor<T, Dimensions, AccessMode> {
        auto sycl_queue = cgh.sycl_queue();
        std::vector<partition_ptr> partitions
            = nd_partition<T, Dimensions>::create_empty_partitions(
                sycl_queue,
                shape_,
                partition_size
            );
        auto device_part_data_ptrs
            = ::sclx::make_shared<raw_partition_data_ptr[]>(
                sycl_queue,
                usm::alloc::device,
                partitions.size()
            );
        auto device_part_write_bits_ptrs
            = ::sclx::make_shared<raw_partition_write_bits_ptr[]>(
                sycl_queue,
                usm::alloc::device,
                partitions.size()
            );
        cgh.register_command_data(device_part_data_ptrs);
        cgh.register_command_data(device_part_write_bits_ptrs);
        cgh.register_command_data(partitions);

        auto cmd_config = cgh.cmd_config();
        std::vector<std::shared_ptr<detail::partition_interface>>
            type_erased_parts;
        for (auto& part : partitions) {
            type_erased_parts.push_back(
                std::static_pointer_cast<detail::partition_interface>(part)
            );
        }
        auto alloc_task = strategy->execute_strategy(
            type_erased_parts,
            std::move(cmd_config)
        );

        auto assign_device_part_ptrs_task = this->assign_device_part_ptrs(
            device_part_data_ptrs,
            device_part_write_bits_ptrs,
            partitions
        );

        auto fetch_partitions_task = this->fetch_partitions(partitions);

        auto push_partitions_task
            = this->push_partitions<AccessMode>(partitions);

        auto fetch_wait_task   = cgh.fetch_wait_task();
        auto command_wait_task = cgh.command_wait_task();
        auto push_wait_task    = cgh.command_wait_task();

        this->configure_task_dependencies<AccessMode>(
            alloc_task,
            assign_device_part_ptrs_task,
            fetch_partitions_task,
            fetch_wait_task,
            command_wait_task,
            push_partitions_task,
            push_wait_task
        );

        accessor<T, Dimensions, AccessMode> acsr{
            device_part_data_ptrs.get(),
            device_part_write_bits_ptrs.get(),
            shape_,
            partitions.front()->shape().size()
        };
        return acsr;
    }

    [[nodiscard]] auto assign_device_part_ptrs(
        ::sclx::shared_ptr<raw_partition_data_ptr[]> device_part_data_ptrs,
        ::sclx::shared_ptr<raw_partition_write_bits_ptr[]>
            device_part_write_bits_ptrs,
        const std::vector<partition_ptr>& partitions
    ) const -> generic_task {
        return create_task(
            [](::sclx::shared_ptr<raw_partition_data_ptr[]>&& device_part_ptrs,
               ::sclx::shared_ptr<raw_partition_write_bits_ptr[]>&&
                   device_part_write_bits_ptrs,
               std::vector<partition_ptr> partitions) {
                std::vector<raw_partition_data_ptr> host_part_data_ptrs;
                std::vector<raw_partition_write_bits_ptr>
                    host_part_write_bits_ptrs;

                for (size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
                    const auto& partition = partitions[p_idx];
                    raw_partition_data_ptr raw_data
                        = static_cast<raw_partition_data_ptr>(
                            partition->pointer()
                        );
                    raw_partition_write_bits_ptr raw_write_bits
                        = partition->write_bits_pointer();
                    host_part_data_ptrs.push_back(raw_data);
                    host_part_write_bits_ptrs.push_back(raw_write_bits);
                }

                sycl::queue native_queue = partitions.front()->queue();
                auto data_copy_event     = native_queue.memcpy(
                    device_part_ptrs.get(),
                    host_part_data_ptrs.data(),
                    host_part_data_ptrs.size() * sizeof(raw_partition_data_ptr)
                );
                auto write_bits_copy_event = native_queue.memcpy(
                    device_part_write_bits_ptrs.get(),
                    host_part_write_bits_ptrs.data(),
                    host_part_write_bits_ptrs.size()
                        * sizeof(raw_partition_write_bits_ptr)
                );
                data_copy_event.wait();
            },
            std::move(device_part_data_ptrs),
            std::move(device_part_write_bits_ptrs),
            partitions
        );
    }

    [[nodiscard]] auto fetch_partitions(std::vector<partition_ptr> partitions
    ) const -> generic_task {
        auto partition_metadata_view
            = partition_metadata_.template get_view<access_mode::write>(
                std::defer_lock
            );
        return create_task(
            [](std::vector<partition_ptr> dst_partitions,
               concurrent_view<partition_metadata> src_partitions_view) {
                src_partitions_view.lock();
                if (src_partitions_view->primary_partitions_.empty()) {
                    src_partitions_view->primary_partitions_ = dst_partitions;
                    return;
                }

                std::vector<sycl::event> fetch_events;

                std::transform(
                    dst_partitions.begin(),
                    dst_partitions.end(),
                    src_partitions_view->primary_partitions_.begin(),
                    std::back_inserter(fetch_events),
                    [](auto& dst_part, const auto& src_part) {
                        return src_part->copy_to(dst_part);
                    }
                );

                std::transform(
                    dst_partitions.begin(),
                    dst_partitions.end(),
                    src_partitions_view->primary_partitions_.begin(),
                    src_partitions_view->primary_partitions_.begin(),
                    [](auto& dst_part, const auto& src_part) {
                        if (dst_part->pointer() != nullptr) {
                            return dst_part;
                        } else {
                            return src_part;
                        }
                    }
                );

                if (src_partitions_view->potentially_active_partitions_.size()
                    != src_partitions_view->primary_partitions_.size()) {
                    src_partitions_view->potentially_active_partitions_.resize(
                        src_partitions_view->primary_partitions_.size(),
                        {}
                    );
                }

                std::transform(
                    dst_partitions.begin(),
                    dst_partitions.end(),
                    src_partitions_view->potentially_active_partitions_.begin(),
                    src_partitions_view->potentially_active_partitions_.begin(),
                    [](auto& dst_part, auto& active_parts) {
                        if (dst_part->pointer() != nullptr) {
                            active_parts.push_back(dst_part);
                        }
                        return active_parts;
                    }
                );

                std::for_each(
                    fetch_events.begin(),
                    fetch_events.end(),
                    [](auto& event) { event.wait(); }
                );
            },
            partitions,
            std::move(partition_metadata_view)
        );
    }

    template<access_mode AccessMode>
    [[nodiscard]] auto push_partitions(std::vector<partition_ptr> partitions
    ) const -> generic_task {
        if constexpr (AccessMode == access_mode::read) {
            return create_task([]() {});
        }
        auto partition_metadata_view
            = partition_metadata_.template get_view<access_mode::write>(
                std::defer_lock
            );
        return create_task(
            [](std::vector<partition_ptr> src_partitions,
               concurrent_view<partition_metadata> dst_partitions_view) {
                dst_partitions_view.lock();

                std::vector<sycl::event> push_events;

                std::transform(
                    src_partitions.begin(),
                    src_partitions.end(),
                    dst_partitions_view->primary_partitions_.begin(),
                    std::back_inserter(push_events),
                    [](auto& src_part, const auto& dst_part) {
                        auto src_queue = src_part->queue();
                        auto dst_queue = dst_part->queue();
                        if (dst_queue.get_device() == src_queue.get_device()) {
                            return src_part->copy_to(dst_part);
                        }

                        auto src_part_on_dst_device
                            = nd_partition<T, Dimensions>::
                                create_empty_partition(
                                    dst_queue,
                                    src_part->shape()
                                );
                        src_part_on_dst_device->allocate();
                        auto copy_event
                            = src_part->copy_to(src_part_on_dst_device);

                        auto write_bits_of_src
                            = src_part_on_dst_device->write_bits_pointer();
                        auto src_data = static_cast<raw_partition_data_ptr>(
                            src_part_on_dst_device->pointer()
                        );
                        auto part_size = src_part_on_dst_device->shape().size();
                        auto dst_data  = static_cast<raw_partition_data_ptr>(
                            dst_part->pointer()
                        );
                        auto write_bits_of_dst = dst_part->write_bits_pointer();
                        return dst_queue.submit([&](sycl::handler& cgh) {
                            cgh.depends_on(copy_event);
                            cgh.parallel_for(
                                sycl::range<>(part_size),
                                [=](sycl::id<> idx) {
                                    if (write_bits_of_src[idx] == 1) {
                                        dst_data[idx] = src_data[idx];
                                    }
                                    write_bits_of_dst[idx] = 0;
                                }
                            );
                        });
                    }
                );

                std::for_each(
                    push_events.begin(),
                    push_events.end(),
                    [](auto& event) { event.wait(); }
                );

                push_events.clear();

                std::transform(
                    dst_partitions_view->primary_partitions_.begin(),
                    dst_partitions_view->primary_partitions_.end(),
                    dst_partitions_view->potentially_active_partitions_.begin(),
                    dst_partitions_view->potentially_active_partitions_.begin(),
                    [&push_events](auto& primary_part, auto& active_parts) {
                        std::erase_if(
                            active_parts,
                            [&](weak_partition_ptr active_part) {
                                auto active_part_lock = active_part.lock();
                                if (active_part_lock != nullptr) {
                                    push_events.push_back(
                                        primary_part->copy_to(active_part_lock)
                                    );
                                    return false;
                                }
                                return true;
                            }
                        );
                        return active_parts;
                    }
                );

                std::for_each(
                    push_events.begin(),
                    push_events.end(),
                    [](auto& event) { event.wait(); }
                );
            },
            partitions,
            std::move(partition_metadata_view)
        );
    }

    template<access_mode AccessMode>
    void configure_task_dependencies(
        generic_task& alloc_task,
        generic_task& assign_device_part_ptrs_task,
        generic_task& fetch_partitions_task,
        generic_task& fetch_wait_task,
        generic_task& command_wait_task,
        generic_task& push_partitions_task,
        generic_task& push_wait_task
    ) const {
        {
            auto queue_commands_view
                = queue_commands_.template get_view<access_mode::write>();
            std::erase_if(
                queue_commands_view->read_queue_commands_,
                [&fetch_partitions_task, &push_wait_task](auto& task) {
                    if (push_wait_task != task
                        && AccessMode != access_mode::read) {
                        task.add_dependent_task(fetch_partitions_task);
                    }
                    return task.has_completed();
                }
            );

            std::erase_if(
                queue_commands_view->write_queue_commands_,
                [&fetch_partitions_task, &push_wait_task](auto& task) {
                    if (push_wait_task != task
                        && AccessMode != access_mode::read) {
                        task.add_dependent_task(fetch_partitions_task);
                    }
                    return task.has_completed();
                }
            );

            if constexpr (AccessMode == access_mode::read) {
                queue_commands_view->read_queue_commands_.push_back(
                    push_wait_task
                );
            } else {
                queue_commands_view->write_queue_commands_.push_back(
                    push_wait_task
                );
            }
        }

        alloc_task.add_dependent_task(assign_device_part_ptrs_task);
        alloc_task.launch();

        assign_device_part_ptrs_task.add_dependent_task(fetch_partitions_task);
        assign_device_part_ptrs_task.launch();

        fetch_partitions_task.add_dependent_task(fetch_wait_task);
        fetch_partitions_task.launch();

        command_wait_task.add_dependent_task(push_partitions_task);

        push_partitions_task.add_dependent_task(push_wait_task);
        push_partitions_task.launch();
    }

    sycl::range<Dimensions> shape_;
    concurrent_guard<queue_commands> queue_commands_;
    concurrent_guard<partition_metadata> partition_metadata_;
};
}  // namespace detail

struct default_access_strategy {
    template<class T, int Dimensions>
    friend struct buffer;

    struct strat_imp : public access_strategy {
        using command_config = access_strategy::command_config;
        [[nodiscard]] auto execute_strategy(
            std::vector<std::shared_ptr<detail::partition_interface>>&
                partitions,
            std::future<std::unique_ptr<command_config>> /*cmd_config*/
        ) -> generic_task override {
            return create_task(
                [](const std::decay_t<decltype(partitions)>& partitions) {
                    for (const auto& partition : partitions) {
                        partition->allocate();
                    }
                },
                partitions
            );
        }
    };
    auto create() -> std::unique_ptr<access_strategy> {
        return std::move(strat_imp_);
    }

    std::unique_ptr<access_strategy> strat_imp_ = std::make_unique<strat_imp>();
};

template<class T, int Dimensions>
struct buffer {
  public:
    explicit buffer(const sycl::range<Dimensions>& range)
        : impl_(std::shared_ptr<detail::buffer_imp<T, Dimensions>>(
              new detail::buffer_imp<T, Dimensions>(range)
          )) {}

    template<
        access_mode AccessMode,
        class StrategySpec = default_access_strategy>
    auto get_access(
        command_handler& cgh,
        StrategySpec access_strategy = default_access_strategy{}
    ) -> accessor<T, Dimensions, AccessMode> {
        if constexpr (AccessMode == access_mode::read) {
            return impl_->get_read_access(cgh, access_strategy.create());
        } else {
            return impl_->get_write_access(cgh, access_strategy.create());
        }
    }

    // private:
    std::shared_ptr<detail::buffer_imp<T, Dimensions>> impl_;
};

template<class FunctionType>
auto queue::submit(FunctionType submission) {
    auto fetch_wait_task  = create_task([]() {});
    auto fetch_wait_event = future_to_event(
        sub_queues.front(),
        fetch_wait_task.get_future().share()
    );

    std::promise<std::vector<sycl::event>> native_events_promise;
    auto command_wait_task = create_task(
        [](std::future<std::vector<sycl::event>> native_events_future) {
            auto native_events = native_events_future.get();
            for (auto& event : native_events) {
                event.wait();
            }
        },
        native_events_promise.get_future()
    );

    std::promise<std::unique_ptr<global_metadata>> global_metadata_promise;
    auto push_wait_task = create_task(
        [](std::future<std::unique_ptr<global_metadata>> global_metadata_future
        ) {
            auto global_metadata = global_metadata_future.get();
            global_metadata.reset();
        },
        global_metadata_promise.get_future()
    );

    auto global_metadata_ptr
        = std::unique_ptr<global_metadata>(new global_metadata{
            sub_command_weights,
            sub_queues,
            {},
            fetch_wait_task,
            command_wait_task,
            push_wait_task,
            0
        });
    auto* global_metadata_raw_ptr = global_metadata_ptr.get();
    global_metadata_promise.set_value(std::move(global_metadata_ptr));

    std::vector<sycl::event> sub_cmd_events;
    for (uint sub_cmd_idx = 0; sub_cmd_idx < sub_queues.size(); ++sub_cmd_idx) {
        auto event
            = sub_queues[sub_cmd_idx].submit([&](sycl::handler& native_cgh) {
                  native_cgh.depends_on(fetch_wait_event);
                  command_handler cgh{
                      global_metadata_raw_ptr,
                      sub_cmd_idx,
                      &native_cgh
                  };
                  submission(cgh);
              });
        sub_cmd_events.push_back(event);
    }
    native_events_promise.set_value(std::move(sub_cmd_events));

    return future_to_event(
        sub_queues.front(),
        push_wait_task.get_future().share()
    );
}

}  // namespace sclx
