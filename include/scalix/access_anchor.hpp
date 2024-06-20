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
#include "detail/page_data.hpp"
#include "typed_task.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <future>
#include <hipSYCL/sycl/event.hpp>
#include <hipSYCL/sycl/queue.hpp>
#include <memory>
#include <scalix/accessor.hpp>
#include <scalix/concurrent_guard.hpp>
#include <scalix/defines.hpp>
#include <scalix/generic_task.hpp>
#include <scalix/pointers.hpp>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sclx {

class object_id {
  public:
    auto operator==(const object_id&) const -> bool = default;
    auto operator!=(const object_id&) const -> bool = default;

  private:
    void* id_{nullptr};
};

struct access_strategy_interface {
    enum access_locale : std::uint8_t { host, device };
    using access_marker     = signed char;
    using access_marker_ptr = access_marker*;

    virtual void init_anchor_for_device(
        page_count_t page_count,
        const sycl::queue& device_queue
    ) = 0;

    [[nodiscard]] virtual auto
    get_page_ptrs(const sycl::device& device, access_locale locale) const
        -> page_ptr_t* = 0;

    [[nodiscard]] virtual auto
    get_shared_page_access_markers(const sycl::device& device
    ) const -> access_marker_ptr* = 0;

    virtual auto ready_accessor(
        sycl::queue device_queue,
        generic_task command_task,
        sycl::range<1> global_range,
        sycl::range<1> local_range,
        sycl::id<1> range_offset
    ) -> generic_task = 0;

    virtual sycl::event make_shared_pages_consistent() = 0;

    virtual ~access_strategy_interface() = default;
};

struct buffer_helper_interface {
    virtual auto update_page_data_for(
        page_index_t page_index,
        shared_ptr<detail::page_data_interface> page
    ) -> sycl::event = 0;
    virtual void
    make_pages_valid(std::vector<shared_ptr<detail::page_data_interface>>& pages
    )                                                                       = 0;
    [[nodiscard]] virtual auto get_number_of_pages() const -> page_count_t  = 0;
    [[nodiscard]] virtual auto get_elements_per_page() const -> std::size_t = 0;
    [[nodiscard]] virtual auto get_element_size() const -> std::size_t      = 0;
    [[nodiscard]] virtual auto get_page_size() const -> page_size_t         = 0;
    virtual auto allocate_page(sycl::device device
    ) -> shared_ptr<detail::page_data_interface>                            = 0;
    virtual ~buffer_helper_interface() = default;
};

template<class T, int Dimensions>
struct buffer_helper_base : buffer_helper_interface {
    virtual auto range() const -> const range<Dimensions>& = 0;
};

template<class T, int Dimensions, page_size_t PageSize>
struct buffer_helper : buffer_helper_base<T, Dimensions> {
    using element_type              = T;
    static constexpr auto page_size = PageSize;
    static constexpr auto elements_per_page
        = (page_size + sizeof(T) - 1) / sizeof(T);
    static constexpr auto element_size = sizeof(T);

    [[nodiscard]] auto get_elements_per_page() const -> std::size_t override {
        return elements_per_page;
    }

    [[nodiscard]] auto get_element_size() const -> std::size_t override {
        return element_size;
    }

    auto allocate_page(sycl::device device
    ) -> shared_ptr<detail::page_data_interface> override {
        auto page_ptr = make_shared<std::byte[]>(
            sycl::queue{device},
            sycl::usm::alloc::device,
            elements_per_page * element_size
        );
        return std::make_shared<detail::page_data<page_size>>(
            page_ptr.get(),
            page_ptr
        );
    }
};

struct access_anchor {
    struct anchor_info {
        struct device_anchor {
            std::vector<shared_ptr<detail::page_data_interface>> pages_;
            std::vector<page_ptr_t> host_page_ptrs_;
            unique_ptr<page_ptr_t[]> device_page_ptrs_;
            std::vector<unique_ptr<access_strategy_interface::access_marker[]>>
                host_page_access_markers_;
            unique_ptr<access_strategy_interface::access_marker_ptr[]>
                shared_page_access_markers_;
        };

        std::unordered_map<sycl::device, device_anchor> device_anchors_;
    };
    std::shared_ptr<anchor_info> info_;

    [[nodiscard]] auto get_page_ptrs(
        const sycl::device& device,
        access_strategy_interface::access_locale locale
        = access_strategy_interface::access_locale::device
    ) const -> page_ptr_t* {
        auto& device_anchor = info_->device_anchors_[device];
        return locale == access_strategy_interface::access_locale::host
                 ? device_anchor.host_page_ptrs_.data()
                 : device_anchor.device_page_ptrs_.get();
    }

    [[nodiscard]] auto get_shared_page_access_markers(const sycl::device& device
    ) const -> access_strategy_interface::access_marker_ptr* {
        return info_->device_anchors_[device].shared_page_access_markers_.get();
    }
};

struct access_strategy_common : access_strategy_interface {
    access_anchor anchor_;

    void init_anchor_for_device(
        page_count_t page_count,
        const sycl::queue& device_queue
    ) override {
        if (anchor_.info_ == nullptr) {
            anchor_.info_ = std::make_shared<access_anchor::anchor_info>();
        }
        auto& device_anchor
            = anchor_.info_->device_anchors_[device_queue.get_device()];
        device_anchor.host_page_ptrs_.resize(page_count);
        device_anchor.device_page_ptrs_ = std::move(make_unique<page_ptr_t[]>(
            device_queue,
            usm::alloc::device,
            page_count
        ));
        device_anchor.host_page_access_markers_.resize(page_count);
        device_anchor.shared_page_access_markers_
            = make_unique<access_marker_ptr[]>(
                device_queue,
                usm::alloc::device,
                page_count
            );
    }

    [[nodiscard]] auto
    get_page_ptrs(const sycl::device& device, access_locale locale) const
        -> page_ptr_t* override {
        return anchor_.get_page_ptrs(device, locale);
    }

    [[nodiscard]] auto get_shared_page_access_markers(const sycl::device& device
    ) const -> access_marker_ptr* override {
        return anchor_.get_shared_page_access_markers(device);
    }

    auto get_primary_queue() const -> sycl::queue {
        return sycl::queue{anchor_.info_->device_anchors_.begin()->first};
    }

    auto get_primary_device() const -> const sycl::device& {
        return anchor_.info_->device_anchors_.begin()->first;
    }

    sycl::event
    make_shared_pages_consistent_impl(buffer_helper_interface* buffer_helper
    ) const {
        auto primary_queue         = get_primary_queue();
        const auto& primary_device = get_primary_device();
        auto elements_per_page     = buffer_helper->get_elements_per_page();
        auto element_size          = buffer_helper->get_element_size();

        std::vector<sycl::event> events;

        // for the pages that are shared between devices, copy the data
        // to page data on the primary device
        using page_mapping_t
            = std::pair<sycl::device, shared_ptr<detail::page_data_interface>>;
        std::vector<std::vector<page_mapping_t>> mapped_peer_page_data(
            anchor_.info_->device_anchors_.begin()->second.pages_.size()
        );
        std::vector<std::shared_future<void>> copy_futures;
        for (auto& [device, device_anchor] : anchor_.info_->device_anchors_) {
            for (page_index_t page_idx = 0;
                 page_idx < mapped_peer_page_data.size();
                 ++page_idx) {
                auto& markers
                    = device_anchor.host_page_access_markers_[page_idx];
                if (markers == nullptr && &device != &primary_device) {
                    continue;
                }

                if (markers == nullptr && &device == &primary_device) {
                    markers = std::move(make_unique<access_marker[]>(
                        primary_queue,
                        usm::alloc::device,
                        elements_per_page
                    ));
                    primary_queue
                        .memset(
                            markers.get(),
                            -1,
                            buffer_helper->get_elements_per_page()
                        )
                        .wait_and_throw();
                }
                auto& peer_data = mapped_peer_page_data[page_idx];
                peer_data.emplace_back(
                    device,
                    buffer_helper->allocate_page(primary_device)
                );
                auto fut = device_anchor.pages_[page_idx]->copy_to(
                    *peer_data.back().second
                );
                copy_futures.push_back(fut.share());
            }
        }

        auto event = primary_queue.submit([&](sycl::handler& cgh) {
            cgh.hipSYCL_enqueue_custom_operation(
                [copy_futures](sycl::interop_handle& h) mutable {
                    std::transform(
                        copy_futures.begin(),
                        copy_futures.end(),
                        copy_futures.begin(),
                        [](auto& fut) {
                            fut.wait();
                            return fut;
                        }
                    );
                }
            );
        });
        events.push_back(event);

        // copy the pointers to the page data and the access markers
        // to an array of pointers allocated on the primary device
        for (page_index_t page_idx = 0; page_idx < mapped_peer_page_data.size();
             ++page_idx) {
            std::vector<page_mapping_t>& peer_data_list
                = mapped_peer_page_data[page_idx];
            const auto number_of_peers = peer_data_list.size();
            auto peer_pages            = make_unique<page_ptr_t[]>(
                primary_queue,
                usm::alloc::device,
                number_of_peers
            );
            auto peer_markers = make_unique<access_marker_ptr[]>(
                primary_queue,
                usm::alloc::device,
                number_of_peers
            );

            auto page_events = std::vector<sycl::event>{events.front()};

            for (int peer_idx = 0; peer_idx < peer_data_list.size();
                 ++peer_idx) {
                auto page_address
                    = peer_data_list[peer_idx].second->page_address();
                event = primary_queue.memcpy(
                    peer_pages.get() + peer_idx,
                    &page_address,
                    sizeof(page_ptr_t),
                    page_events.front()
                );
                page_events.push_back(event);
                auto markers_address
                    = anchor_.info_
                          ->device_anchors_[peer_data_list[peer_idx].first]
                          .host_page_access_markers_[page_idx]
                          .get();
                event = primary_queue.memcpy(
                    peer_markers.get() + peer_idx,
                    &markers_address,
                    sizeof(access_marker_ptr),
                    page_events.front()
                );
                page_events.push_back(event);
            }

            // update primary page with data from peers
            event = primary_queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(page_events);

                auto raw_peer_pages   = peer_pages.get();
                auto raw_peer_markers = peer_markers.get();
                cgh.parallel_for(
                    sycl::range{elements_per_page},
                    [raw_peer_pages,
                     raw_peer_markers,
                     number_of_peers,
                     elements_per_page,
                     element_size](sycl::id<> idx) {
                        auto primary_page_ptr = *raw_peer_pages;
                        auto primary_markers  = *raw_peer_markers;
                        for (int peer_idx = 1; peer_idx < number_of_peers;
                             ++peer_idx) {
                            auto peer_page_ptr = *(raw_peer_pages + peer_idx);
                            auto p_markers     = *(raw_peer_markers + peer_idx);
                            if (primary_page_ptr == nullptr
                                || peer_page_ptr == nullptr) {
                                continue;
                            }
                            if (primary_page_ptr == peer_page_ptr) {
                                continue;
                            }
                            for (int elem = 0; elem < elements_per_page;
                                 ++elem) {
                                if (*(primary_markers + elem) != -1
                                    || *(p_markers + elem) == 0) {
                                    continue;
                                }
                                std::memcpy(
                                    primary_page_ptr + elem * element_size,
                                    peer_page_ptr + elem * element_size,
                                    element_size
                                );
                            }
                        }
                    }
                );
            });
            page_events.push_back(event);

            // copy the updated primary page back to the peers
            for (int peer_idx = 0; peer_idx < number_of_peers; ++peer_idx) {
                page_mapping_t& peer_data_pair = peer_data_list[peer_idx];
                auto& peer_device_anchor
                    = anchor_.info_->device_anchors_[peer_data_pair.first];
                auto source_page      = peer_data_list.front().second;
                auto destination_page = peer_device_anchor.pages_[page_idx];
                event = primary_queue.submit([&](sycl::handler& cgh) {
                    cgh.depends_on(page_events);
                    cgh.hipSYCL_enqueue_custom_operation(
                        [source_page, destination_page](sycl::interop_handle& h
                        ) { source_page->copy_to(*destination_page).wait(); }
                    );
                });
                page_events.push_back(event);
            }

            std::transform(
                page_events.begin(),
                page_events.end(),
                std::back_inserter(events),
                [](const auto& event) { return event; }
            );
        }

        return primary_queue.submit([&events](sycl::handler& cgh) {
            cgh.depends_on(events);
        });
    }
};

struct handler {
    template<int RangeDimensions>
    struct parallel_for_config {
        std::vector<sycl::range<RangeDimensions>> local_ranges_;
        std::vector<sycl::id<RangeDimensions>> range_offsets_;
    };

    template<int RangeDimensions>
    auto get_parallel_for_config(const sycl::range<RangeDimensions>& range
    ) const -> const parallel_for_config<RangeDimensions>& {
        auto global_metadata
            = metadata_->global_metadata_.get_view<access_mode::write>();
        if (global_metadata.access().command_config_ != nullptr) {
            return *std::static_pointer_cast<
                parallel_for_config<RangeDimensions>>(
                global_metadata.access().command_config_
            );
        }

        global_metadata.access().command_config_
            = std::make_shared<parallel_for_config<RangeDimensions>>();
        auto config
            = std::static_pointer_cast<parallel_for_config<RangeDimensions>>(
                global_metadata.access().command_config_
            );
        auto range_dim0            = range[0];
        auto offset_dim0           = decltype(range_dim0){0};
        auto local_ranges_dim0_sum = decltype(range_dim0){0};
        for (double weight : global_metadata.access().weights_) {
            auto local_range_dim0
                = static_cast<decltype(range_dim0)>(range_dim0 * weight);
            sycl::range<RangeDimensions> local_range{range};
            local_range[0] = local_range_dim0;
            sycl::id<RangeDimensions> range_offset;
            range_offset[0] = offset_dim0;
            offset_dim0 += local_range_dim0;
            local_ranges_dim0_sum += local_range_dim0;

            config->local_ranges_.push_back(local_range);
            config->range_offsets_.push_back(range_offset);
        }

        if (config->range_offsets_.back()[0] + config->local_ranges_.back()[0]
            < range[0]) {
            config->local_ranges_.back()[0] += range[0] - local_ranges_dim0_sum;
        }

        return *config;
    }

    template<int RangeDimensions = 1, class Kernel>
    void parallel_for(sycl::range<RangeDimensions> range, Kernel&& kernel) {
        const auto& meta    = metadata_;
        auto& launch_config = get_parallel_for_config(range);
        auto local_range    = launch_config.local_ranges_[meta->device_idx_];
        auto range_offset   = launch_config.range_offsets_[meta->device_idx_];

        auto command_submit_promise = std::promise<void>{};
        auto command_submit_future_ptr
            = new std::future<void>{command_submit_promise.get_future()};
        auto command_submit_task = create_task(
            [](std::promise<void>&& prom) {
                prom.set_value(); },
            std::move(command_submit_promise)
        );

        std::vector<generic_task> command_tasks;
        auto global_metadata
            = meta->global_metadata_.get_view<access_mode::write>();
        for (auto& [unused, strategy] : global_metadata.access().strategies_) {
            auto task = strategy->ready_accessor(
                meta->device_queue_,
                command_submit_task,
                range,
                local_range,
                range_offset
            );
            command_tasks.push_back(task);
        }

        command_submit_task.launch();

        sycl::buffer<void*> command_submit_future_buffer{sycl::range{1}};
        {
            auto acsr = command_submit_future_buffer.get_access<sycl::access_mode::discard_write>();
            acsr[0] = command_submit_future_ptr;
        }
        sycl::event command_submit_event
            = sycl::queue{}.submit([&](sycl::handler& cgh) {
                  auto acsr = command_submit_future_buffer.get_access<sycl::access_mode::read>();
                  cgh.hipSYCL_enqueue_custom_operation(
                      [=](sycl::interop_handle& h) {
                          auto fut_ptr = acsr[0];
                          static_cast<std::future<void>*>(fut_ptr)->wait();
                          delete fut_ptr;
                      }
                  );
              });
        auto command_task = create_task(
            [meta, command_submit_event, &kernel, &local_range, &range_offset]() {
                meta->device_queue_.submit([&](sycl::handler& cgh) {
                    cgh.depends_on(command_submit_event);
                    cgh.parallel_for(
                        local_range,
                        [=](sycl::id<RangeDimensions> idx) {
                            kernel(idx + range_offset);
                        }
                    );
                }).wait_and_throw();
            }
        );
        command_tasks.push_back(command_task);

        auto dummy_task = create_task([] {});
        for (auto& task : command_tasks) {
            task.add_dependent_task(dummy_task);
        }
        void* fut = new std::future<void>{dummy_task.get_future()};

        command_task.launch();
        dummy_task.launch();

        sycl::buffer<void*> fut_buffer{sycl::range{1}};
        {
            auto acsr = fut_buffer.get_access<sycl::access_mode::discard_write>();
            acsr[0] = fut;
        }
        auto command_event = sycl::queue{}.submit([&](sycl::handler& cgh) {
            auto acsr = fut_buffer.get_access<sycl::access_mode::read>();
            cgh.hipSYCL_enqueue_custom_operation(
                [=](sycl::interop_handle& h) {
                    auto fut_ptr = acsr[0];
                    static_cast<std::future<void>*>(fut_ptr)->wait();
                    delete static_cast<std::future<void>*>(fut_ptr);
                }
            );
        });

        metadata_->command_event_ = std::move(command_event);
    }

    void assign_strategy(
        void* buffer_ptr,
        std::shared_ptr<access_strategy_interface> strategy
    ) {
        auto global_metadata
            = metadata_->global_metadata_.get_view<access_mode::write>();
        global_metadata.access().strategies_[buffer_ptr] = std::move(strategy);
    }

    auto get_buffer_access_strategy(void* buffer_ptr
    ) const -> std::shared_ptr<access_strategy_interface>& {
        auto global_metadata
            = metadata_->global_metadata_.get_view<access_mode::write>();
        return global_metadata.access().strategies_[buffer_ptr];
    }

    [[nodiscard]] auto get_command_event() const -> sycl::event {
        return metadata_->command_event_;
    }

    [[nodiscard]] auto device() const -> const sycl::device& {
        return metadata_->device_;
    }

    struct metadata {
        sycl::queue device_queue_;
        sycl::device device_;
        size_t device_idx_;
        sycl::event command_event_;

        struct global_metadata {
            std::vector<double> weights_;
            std::shared_ptr<void> command_config_;
            std::
                unordered_map<void*, std::shared_ptr<access_strategy_interface>>
                    strategies_;
        };

        concurrent_guard<global_metadata> global_metadata_;
    };
    std::shared_ptr<metadata> metadata_;
};

template<class T, int Dimensions>
struct buffer {
    template<page_size_t PageSize>
    struct impl : buffer_helper<T, Dimensions, PageSize> {
        std::vector<std::weak_ptr<detail::page_data_interface>> pages_;
        std::vector<access_anchor> anchors_;
        range<Dimensions> range_;

        void make_pages_valid(
            std::vector<shared_ptr<detail::page_data_interface>>& pages
        ) override {
            std::transform(
                pages_.begin(),
                pages_.end(),
                pages.begin(),
                pages.begin(),
                [](auto& weak_page, const auto& new_page) {
                    auto page = weak_page.lock();
                    if (page == nullptr) {
                        return new_page;
                    }
                    page->copy_to(*new_page).wait();
                    return new_page;
                }
            );
        }

        [[nodiscard]] auto range() const -> const range<Dimensions>& override {
            return range_;
        }

        [[nodiscard]] auto get_page_size() const -> page_size_t override {
            return PageSize;
        }

        auto update_page_data_for(
            page_index_t page_index,
            shared_ptr<detail::page_data_interface> page
        ) -> sycl::event override {
            auto page_data
                = std::static_pointer_cast<detail::page_data<PageSize>>(page);
            this->pages_[page_index] = page;
            std::vector<std::shared_future<void>> events;
            for (auto& anchor : this->anchors_) {
                for (auto& [device, device_anchor] :
                     anchor.info_->device_anchors_) {
                    auto event
                        = page_data->copy_to(*device_anchor.pages_[page_index]);
                    events.push_back(event.share());
                }
            }
            return sycl::queue{}.submit([&events](sycl::handler& cgh) {
                cgh.hipSYCL_enqueue_custom_operation(
                    [events](sycl::interop_handle& h) {
                        for (auto& event : events) {
                            event.wait();
                        }
                    }
                );
            });
        }

        [[nodiscard]] auto
        get_number_of_pages() const -> page_count_t override {
            return this->pages_.size();
        }
    };
    concurrent_guard<buffer_helper_interface> impl_{std::shared_ptr<impl<4096>>{
    }};

    explicit buffer(range<Dimensions> range)
        : impl_{std::static_pointer_cast<buffer_helper_interface>(
              std::make_shared<impl<4096>>()
          )} {
        static_cast<impl<4096>&>(impl_.unsafe_access()).range_ = range;
        auto elements_per_page = impl_.unsafe_access().get_elements_per_page();
        auto num_elements      = range.size();
        auto num_pages
            = (num_elements + elements_per_page - 1) / elements_per_page;
        static_cast<impl<4096>&>(impl_.unsafe_access())
            .pages_.resize(num_pages);
    }

    template<
        access_mode AccessMode = access_mode::read_write,
        class AccessStrategy>
    auto get_access(handler& cgh, const AccessStrategy& strategy) const
        -> accessor<T, Dimensions, AccessMode> {
        auto strategy_ptr
            = cgh.get_buffer_access_strategy(&impl_.unsafe_access());
        if (strategy_ptr == nullptr) {
            strategy_ptr
                = strategy.template get<T, Dimensions, AccessMode>(impl_);
            cgh.assign_strategy(&impl_.unsafe_access(), strategy_ptr);
        }
        strategy_ptr->init_anchor_for_device(
            impl_.unsafe_access().get_number_of_pages(),
            cgh.metadata_->device_queue_
        );
        auto page_ptrs = strategy_ptr->get_page_ptrs(
            cgh.device(),
            access_strategy_interface::access_locale::device
        );
        auto page_offsets
            = strategy_ptr->get_shared_page_access_markers(cgh.device());

        return accessor<T, Dimensions, AccessMode>{
            page_ptrs,
            static_cast<buffer_helper_base<T, Dimensions>&>(impl_.unsafe_access(
                                                            ))
                .range(),
            this->impl_.unsafe_access().get_page_size(),
            page_offsets
        };
    }
};

struct queue {
    template<class Submission>
    auto submit(Submission&& submission) -> sycl::event {
        std::vector<handler> command_handlers(device_weights_.size());
        std::vector<sycl::event> command_events;
        auto command_topology_guard
            = concurrent_guard<handler::metadata::global_metadata>();
        auto& command_topology    = command_topology_guard.unsafe_access();
        command_topology.weights_ = device_weights_;
        for (auto& dqueue : device_queues_) {
            auto idx          = std::distance(&device_queues_.front(), &dqueue);
            auto& handler     = command_handlers[idx];
            handler.metadata_ = std::make_shared<handler::metadata>();
            handler.metadata_->device_queue_ = dqueue;
            handler.metadata_->device_       = dqueue.get_device();
            handler.metadata_->device_idx_
                = std::distance(&device_queues_.front(), &dqueue);
            handler.metadata_->global_metadata_ = command_topology_guard;
            submission(handler);
            command_events.push_back(handler.get_command_event());
        }

        auto global_command_event = device_queues_.front().submit(
            [&command_events, command_handlers](sycl::handler& cgh) {
                cgh.depends_on(command_events);
//                auto primary_handler = command_handlers.front();
//                cgh.hipSYCL_enqueue_custom_operation(
//                    [primary_handler
//                     = std::move(primary_handler)](sycl::interop_handle& h) {
//                        std::vector<sycl::event> finalization_events;
//                        auto global_metadata
//                            = primary_handler.metadata_->global_metadata_
//                                  .get_view<access_mode::read>();
//                        for (auto& [unused, strategy] :
//                             global_metadata.access().strategies_) {
//                            finalization_events.push_back(
//                                strategy->make_shared_pages_consistent()
//                            );
//                        }
//
//                        for (auto& event : finalization_events) {
//                            event.wait_and_throw();
//                        }
//                    }
//                );
            }
        );

        return global_command_event;
    }

    std::vector<sycl::queue> device_queues_;
    std::vector<double> device_weights_;
};

struct default_access_strategy {
    template<access_mode AccessMode>
    struct impl : access_strategy_common {
        using buffer_view_t = std::decay_t<
            decltype(std::declval<concurrent_guard<buffer_helper_interface>>()
                         .get_view<AccessMode>())>;
        auto ready_accessor(
            sycl::queue device_queue,
            generic_task command_task,
            sycl::range<1> global_range,
            sycl::range<1> local_range,
            sycl::id<1> range_offset
        ) -> generic_task override {
            if (shared_view_ == nullptr) {
                shared_view_ = std::make_shared<buffer_view_t>(
                    std::move(buffer_helper_.get_view<AccessMode>())
                );
            }
            auto& shared_view = shared_view_;
            auto prepare_task
                = create_task([]() {});

            auto buffer_ptr = &(shared_view.get()->access());
            std::vector<std::shared_ptr<detail::page_data_interface>> pages(
                buffer_ptr->get_number_of_pages(),
                nullptr
            );
            auto device = device_queue.get_device();
            std::transform(
                pages.begin(),
                pages.end(),
                pages.begin(),
                [&](auto& page) { return buffer_ptr->allocate_page(device); }
            );
            buffer_ptr->make_pages_valid(pages);
            auto& device_anchor
                = this->anchor_.info_
                      ->device_anchors_[device_queue.get_device()];
            std::transform(
                pages.begin(),
                pages.end(),
                device_anchor.host_page_ptrs_.begin(),
                [](auto& page) {
                    return const_cast<page_ptr_t>(page->page_address());
                }
            );
            device_queue
                .memcpy(
                    device_anchor.device_page_ptrs_.get(),
                    device_anchor.host_page_ptrs_.data(),
                    device_anchor.host_page_ptrs_.size() * sizeof(page_ptr_t)
                )
                .wait_and_throw();
            std::transform(
                pages.begin(),
                pages.end(),
                device_anchor.host_page_access_markers_.begin(),
                [&](auto& markers) {
                    auto page_markers = make_unique<access_marker[]>(
                        device_queue,
                        usm::alloc::device,
                        buffer_ptr->get_elements_per_page()
                    );
                    return std::move(page_markers);
                }
            );
            {
                std::vector<sycl::event> memset_events;
                std::transform(
                    device_anchor.host_page_access_markers_.begin(),
                    device_anchor.host_page_access_markers_.end(),
                    std::back_inserter(memset_events),
                    [&](auto& markers) {
                        return device_queue.memset(
                            markers.get(),
                            0,
                            sizeof(access_marker)
                                * buffer_ptr->get_elements_per_page()
                        );
                    }
                );
                std::transform(
                    memset_events.begin(),
                    memset_events.end(),
                    memset_events.begin(),
                    [](auto& event) {
                        event.wait_and_throw();
                        return event;
                    }
                );
            }
            std::vector<void*> raw_markers;
            std::transform(
                device_anchor.host_page_access_markers_.begin(),
                device_anchor.host_page_access_markers_.end(),
                std::back_inserter(raw_markers),
                [](auto& markers) { return markers.get(); }
            );
            device_queue
                .memcpy(
                    device_anchor.shared_page_access_markers_.get(),
                    raw_markers.data(),
                    raw_markers.size() * sizeof(access_marker_ptr)
                )
                .wait_and_throw();

            device_anchor.pages_ = std::move(pages);
            prepare_task.add_dependent_task(command_task);
            prepare_task.launch();

            if constexpr (AccessMode == access_mode::read) {
                return prepare_task;
            }
            auto finalization_task = create_task([]() {
                //                auto buffer_ptr =
                //                &shared_view.get()->access(); auto&
                //                device_anchor
                //                    = this->anchor_.info_
                //                          ->device_anchors_[device_queue.get_device()];
                //                std::vector<sycl::event> update_events;
                //                for (auto& page : device_anchor.pages_) {
                //                    auto page_index =
                //                    static_cast<page_index_t>(
                //                        std::distance(&device_anchor.pages_.front(),
                //                        &page)
                //                    );
                //                    auto event
                //                        =
                //                        buffer_ptr->update_page_data_for(page_index,
                //                        page);
                //                    update_events.push_back(event);
                //                }
                //                for (auto& event : update_events) {
                //                    event.wait_and_throw();
                //                }
            });
            std::vector<sycl::event> update_events;
            for (auto& page : device_anchor.pages_) {
                auto page_index = static_cast<page_index_t>(
                    std::distance(&device_anchor.pages_.front(), &page)
                );
                auto event = buffer_ptr->update_page_data_for(page_index, page);
                update_events.push_back(event);
            }
            for (auto& event : update_events) {
                event.wait_and_throw();
            }
            prepare_task.add_dependent_task(finalization_task);
            finalization_task.launch();
            return finalization_task;
        }

        auto make_shared_pages_consistent() -> sycl::event override {
            if constexpr (AccessMode == access_mode::read) {
                return {};
            }
            std::promise<sycl::event> promise;
            auto future = promise.get_future();
            auto task
                = create_task([*this, promise = std::move(promise)] mutable {
                      auto event = this->make_shared_pages_consistent_impl(
                          &(this->shared_view_.get()->access())
                      );
                      promise.set_value(event);
                  });
            task.launch();
            return future.get();
        }

        impl(
            concurrent_guard<buffer_helper_interface> buffer_helper,
            std::shared_ptr<buffer_view_t> shared_view
        )
            : buffer_helper_{buffer_helper},
              shared_view_{shared_view} {}

        impl(const impl&)                    = default;
        impl(impl&&)                         = default;
        auto operator=(const impl&) -> impl& = default;
        auto operator=(impl&&) -> impl&      = default;

        concurrent_guard<buffer_helper_interface> buffer_helper_{
            std::shared_ptr<buffer_helper_interface>{}
        };
        std::shared_ptr<buffer_view_t> shared_view_;
    };

    template<class, int, access_mode AccessMode>
    auto get(concurrent_guard<buffer_helper_interface> buffer
    ) const -> std::shared_ptr<impl<AccessMode>> {
        return std::make_shared<impl<AccessMode>>(
            typename default_access_strategy::impl<AccessMode>{buffer, nullptr}
        );
    }
};

}  // namespace sclx
