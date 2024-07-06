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
#include <source_location>
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

struct access_anchor;

struct access_strategy_interface {
    enum access_locale : std::uint8_t { host, device };
    using access_marker     = signed char;
    using access_marker_ptr = access_marker*;

    virtual void release() = 0;

    virtual auto get_anchor() const -> std::shared_ptr<access_anchor> = 0;

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
    virtual void register_anchor(std::shared_ptr<access_anchor> anchor)     = 0;
    virtual ~buffer_helper_interface() = default;
};

template<class T, int Dimensions>
struct host_accessor {
    std::vector<std::remove_const_t<T>> data_;
    std::shared_ptr<void> buffer_handle_;
};

template<class T, int Dimensions>
struct buffer_helper_base : buffer_helper_interface {
    virtual auto range() const -> const range<Dimensions>& = 0;

    virtual auto get_host_access() const -> host_accessor<T, Dimensions> = 0;
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
        auto device_queue = sycl::queue{device};
        auto page_ptr     = make_shared<std::byte[]>(
            device_queue,
            sycl::usm::alloc::device,
            elements_per_page * element_size
        );
        return std::make_shared<detail::page_data<page_size>>(
            page_ptr.get(),
            std::reinterpret_pointer_cast<
                typename detail::page_data<page_size>::alloc_handle_t>(page_ptr
            ),
            device_queue
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
    std::unique_ptr<anchor_info> info_;

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
    std::shared_ptr<access_anchor> anchor_;

    auto get_anchor() const -> std::shared_ptr<access_anchor> override {
        return anchor_;
    }

    void init_anchor_for_device(
        page_count_t page_count,
        const sycl::queue& device_queue
    ) override {
        if (anchor_ == nullptr) {
            anchor_        = std::make_shared<access_anchor>();
            anchor_->info_ = std::make_unique<access_anchor::anchor_info>();
        }
        auto& device_anchor
            = anchor_->info_->device_anchors_[device_queue.get_device()];
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
        return anchor_->get_page_ptrs(device, locale);
    }

    [[nodiscard]] auto get_shared_page_access_markers(const sycl::device& device
    ) const -> access_marker_ptr* override {
        return anchor_->get_shared_page_access_markers(device);
    }

    auto get_primary_queue() const -> sycl::queue {
        return sycl::queue{anchor_->info_->device_anchors_.begin()->first};
    }

    auto get_primary_device() const -> const sycl::device& {
        return anchor_->info_->device_anchors_.begin()->first;
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
            anchor_->info_->device_anchors_.begin()->second.pages_.size()
        );
        for (page_index_t page_idx = 0; page_idx < mapped_peer_page_data.size();
             ++page_idx) {
            std::vector<sycl::event> resident_events;
            for (auto& [device, device_anchor] :
                 anchor_->info_->device_anchors_) {
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
                auto event = device_anchor.pages_[page_idx]->copy_to(
                    peer_data.back().second
                );
                resident_events.push_back(event);
            }

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

            std::vector<sycl::event> peer_access_info_events;
            for (int peer_idx = 1; peer_idx < peer_data_list.size();
                 ++peer_idx) {
                auto page_address
                    = peer_data_list[peer_idx].second->page_address();
                auto event = primary_queue.memcpy(
                    peer_pages.get() + peer_idx,
                    &page_address,
                    sizeof(page_ptr_t),
                    resident_events
                );
                event.wait_and_throw();
                peer_access_info_events.push_back(event);
                auto markers_address
                    = anchor_->info_
                          ->device_anchors_[peer_data_list[peer_idx].first]
                          .host_page_access_markers_[page_idx]
                          .get();
                event = primary_queue.memcpy(
                    peer_markers.get() + peer_idx,
                    &markers_address,
                    sizeof(access_marker_ptr),
                    resident_events
                );
                event.wait_and_throw();
                peer_access_info_events.push_back(event);
            }

            // update primary page with data from peers

            primary_queue
                .submit([&](sycl::handler& cgh) {
                    cgh.depends_on(peer_access_info_events);

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
                                auto peer_page_ptr
                                    = *(raw_peer_pages + peer_idx);
                                auto p_markers = *(raw_peer_markers + peer_idx);
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
                })
                .wait_and_throw();

            // copy the updated primary page back to the peers
            std::vector<sycl::event> primary_to_peer_events;
            for (int peer_idx = 1; peer_idx < number_of_peers; ++peer_idx) {
                page_mapping_t& peer_data_pair = peer_data_list[peer_idx];
                auto& peer_device_anchor
                    = anchor_->info_->device_anchors_[peer_data_pair.first];
                auto source_page      = peer_data_list.front().second;
                auto destination_page = peer_device_anchor.pages_[page_idx];
                auto event            = source_page->copy_to(destination_page);
                primary_to_peer_events.push_back(event);
            }

            std::transform(
                primary_to_peer_events.begin(),
                primary_to_peer_events.end(),
                std::back_inserter(events),
                [](const sycl::event& event) { return event; }
            );
        }

        auto combined_event = primary_queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(events);
            cgh.AdaptiveCpp_enqueue_custom_operation([](sycl::interop_handle&) {

            });
        });
        combined_event.wait_and_throw();
        return combined_event;
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
            = metadata_->global_metadata_.get_view<access_mode::write>(
                std::source_location::current()
            );
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
    void parallel_for(sycl::range<RangeDimensions> range, Kernel kernel) {
        const auto& meta    = metadata_;
        auto& launch_config = get_parallel_for_config(range);
        auto local_range    = launch_config.local_ranges_[meta->device_idx_];
        auto range_offset   = launch_config.range_offsets_[meta->device_idx_];

        auto command_submit_promise = std::promise<void>{};
        auto command_submit_future_ptr
            = new std::future<void>{command_submit_promise.get_future()};
        auto command_submit_task = create_task(
            [](std::promise<void>&& prom) { prom.set_value(); },
            std::move(command_submit_promise)
        );

        std::vector<generic_task> command_tasks;
        auto global_metadata = meta->global_metadata_.unsafe_access();
        for (auto& [unused, strategy] : global_metadata.strategies_) {
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
            auto acsr = command_submit_future_buffer
                            .get_access<sycl::access_mode::discard_write>();
            acsr[0] = command_submit_future_ptr;
        }
        sycl::event command_submit_event
            = this->metadata_->device_queue_.submit([&](sycl::handler& cgh) {
                  auto acsr = command_submit_future_buffer
                                  .get_access<sycl::access_mode::read>();
                  cgh.AdaptiveCpp_enqueue_custom_operation(
                      [=](sycl::interop_handle& h) {
                          auto fut_ptr = acsr[0];
                          static_cast<std::future<void>*>(fut_ptr)->wait();
                          delete fut_ptr;
                      }
                  );
              });
        auto command_task = create_task(
            [command_submit_event, &kernel, &local_range, &range_offset](
                decltype(meta) meta
            ) {
                meta->device_queue_
                    .submit([&](sycl::handler& cgh) {
                        cgh.depends_on(command_submit_event);
                        cgh.parallel_for(
                            local_range,
                            [=](sycl::id<RangeDimensions> idx) {
                                kernel(idx + range_offset);
                            }
                        );
                    })
                    .wait_and_throw();
            },
            meta
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
            auto acsr
                = fut_buffer.get_access<sycl::access_mode::discard_write>();
            acsr[0] = fut;
        }
        auto command_event
            = this->metadata_->device_queue_.submit([&](sycl::handler& cgh) {
                  auto acsr = fut_buffer.get_access<sycl::access_mode::read>();
                  cgh.AdaptiveCpp_enqueue_custom_operation(
                      [=](sycl::interop_handle& h) {
                          auto fut_ptr = acsr[0];
                          static_cast<std::future<void>*>(fut_ptr)->wait();
                          delete static_cast<std::future<void>*>(fut_ptr);
                      }
                  );
              });

        metadata_->command_event_ = command_event;
    }

    void assign_strategy(
        void* buffer_ptr,
        std::shared_ptr<access_strategy_interface> strategy
    ) {
        auto global_metadata
            = metadata_->global_metadata_.get_view<access_mode::write>(
                std::source_location::current()
            );
        global_metadata.access().strategies_[buffer_ptr] = std::move(strategy);
    }

    auto get_buffer_access_strategy(void* buffer_ptr
    ) const -> std::shared_ptr<access_strategy_interface> {
        auto global_metadata
            = metadata_->global_metadata_.get_view<access_mode::read>(
                std::source_location::current()
            );
        if (global_metadata.access().strategies_.count(buffer_ptr) == 0) {
            return {nullptr};
        }
        return global_metadata.access().strategies_.at(buffer_ptr);
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
        std::vector<std::shared_ptr<access_anchor>> anchors_;
        range<Dimensions> range_;

        void register_anchor(std::shared_ptr<access_anchor> anchor) override {
            anchors_.push_back(anchor);
        }

        void make_pages_valid(
            std::vector<shared_ptr<detail::page_data_interface>>& pages
        ) override {
            std::vector<sycl::event> events;
            std::transform(
                pages_.begin(),
                pages_.end(),
                pages.begin(),
                std::back_inserter(events),
                [](auto& weak_page, const auto& new_page) {
                    auto page = weak_page.lock();
                    if (weak_page.expired() || page == nullptr) {
                        return sycl::event{};
                    }
                    return page->copy_to(new_page);
                }
            );
            for (auto& event : events) {
                event.wait_and_throw();
            }
            std::transform(
                pages_.begin(),
                pages_.end(),
                pages.begin(),
                pages_.begin(),
                [](auto& weak_page, const auto& new_page) {
                    if (new_page == nullptr) {
                        return weak_page;
                    }
                    return std::weak_ptr<detail::page_data_interface>{new_page};
                }
            );
        }

        auto get_host_access() const -> host_accessor<T, Dimensions> override {
            auto num_elements      = range_.size();
            auto elements_per_page = this->get_elements_per_page();
            auto num_pages
                = (num_elements + elements_per_page - 1) / elements_per_page;
            auto page_size = this->get_page_size();
            host_accessor<T, Dimensions> accessor;
            accessor.data_.resize(num_elements);
            for (auto& page_ptr : this->pages_) {
                auto page_ptr_locked = page_ptr.lock();
                if (page_ptr.expired() || page_ptr_locked == nullptr) {
                    continue;
                }
                sycl::queue page_queue{page_ptr_locked->device_queue()};
                auto host_page_ptr = ::sclx::make_unique<byte[]>(
                    page_queue,
                    usm::alloc::host,
                    page_size
                );
                page_ptr_locked->copy_to(page_queue, host_page_ptr.get())
                    .wait_and_throw();
                auto host_page_ptr_cast
                    = reinterpret_cast<T*>(host_page_ptr.get());

                auto page_idx = std::distance(&this->pages_.front(), &page_ptr);
                auto element_idx = page_idx * elements_per_page;
                std::memcpy(
                    accessor.data_.data() + element_idx,
                    host_page_ptr.get(),
                    std::min(elements_per_page, num_elements - element_idx)
                        * this->get_element_size()
                );
            }

            return accessor;
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
            std::vector<sycl::event> events;
            for (auto& anchor : this->anchors_) {
                for (auto& [device, device_anchor] :
                     anchor->info_->device_anchors_) {
                    auto event
                        = page_data->copy_to(*device_anchor.pages_[page_index]);
                    events.push_back(event);
                }
            }
            auto event = page->device_queue().submit([&events](sycl::handler& cgh) {
                cgh.depends_on(events);
                cgh.AdaptiveCpp_enqueue_custom_operation(
                    [](const sycl::interop_handle& h) {}
                );
            });
            event.wait_and_throw();
            return event;
        }

        [[nodiscard]] auto
        get_number_of_pages() const -> page_count_t override {
            return this->pages_.size();
        }
    };
    concurrent_guard<buffer_helper_interface> impl_{std::shared_ptr<impl<4096>>{
    }};

    template<access_mode AccessMode = access_mode::read_write>
    auto get_access() const
        -> host_accessor<
            std::conditional_t<AccessMode == access_mode::read, const T, T>,
            Dimensions> {
        static constexpr auto mode = AccessMode;
        using view_type_param      = std::conditional_t<
                 mode == access_mode::read,
                 const buffer_helper_interface,
                 buffer_helper_interface>;
        auto view = impl_.get_view<AccessMode>(std::source_location::current());
        auto view_ptr
            = std::make_shared<concurrent_view<view_type_param>>(std::move(view)
            );
        using upcast_type = std::conditional_t<
            mode == access_mode::read,
            const buffer_helper_base<T, Dimensions>,
            buffer_helper_base<T, Dimensions>>;
        auto acsr
            = static_cast<upcast_type&>(view_ptr->access()).get_host_access();
        acsr.buffer_handle_ = view_ptr;

        return reinterpret_cast<host_accessor<
            std::conditional_t<AccessMode == access_mode::read, const T, T>,
            Dimensions>&>(acsr);
    }

    explicit buffer(range<Dimensions> range)
        : impl_{ std::make_shared<impl<4096>>()} {
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
        impl_.unsafe_access().register_anchor(strategy_ptr->get_anchor());
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

        auto global_command_task = create_task(
            [](handler primary_handler,
               sycl::queue primary_queue,
               std::vector<sycl::event> cmd_events) {
                for (auto& event : cmd_events) {
                    event.wait_and_throw();
                }
                std::vector<sycl::event> finalization_events;
                auto global_metadata
                    = primary_handler.metadata_->global_metadata_.unsafe_access(
                    );
                for (auto& [unused, strategy] : global_metadata.strategies_) {
                    finalization_events.push_back(
                        strategy->make_shared_pages_consistent()
                    );
                }

                return finalization_events;
            },
            command_handlers.front(),
            device_queues_.front(),
            command_events
        );
        auto fut = global_command_task.get_future();
        global_command_task.launch();
        auto finalization_events = fut.get();

        auto global_command_event
            = device_queues_.front().submit([&finalization_events](sycl::handler& cgh) {
                  cgh.depends_on(finalization_events);
                  cgh.AdaptiveCpp_enqueue_custom_operation(
                      [](const sycl::interop_handle& h) {}
                  );
              });

//        auto task = create_task([](sycl::event event, std::vector<handler> handlers) {
//            event.wait();
//            for (auto& handler : handlers) {
//                for (auto& strategy : handler.metadata_->global_metadata_.unsafe_access().strategies_) {
//                    strategy.second->release();
//                }
//            }
//        }, global_command_event, command_handlers);
//        task.launch();

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
                         .get_view<AccessMode>(std::source_location::current())
            )>;

        void release() {
            shared_view_.reset();
        }

        auto ready_accessor(
            sycl::queue device_queue,
            generic_task command_task,
            sycl::range<1> global_range,
            sycl::range<1> local_range,
            sycl::id<1> range_offset
        ) -> generic_task override {
            if (shared_view_ == nullptr) {
                shared_view_ = std::make_shared<buffer_view_t>(
                    std::move(buffer_helper_.get_view<AccessMode>(
                        std::source_location::current()
                    ))
                );
            }
            auto& shared_view = shared_view_;
            auto prepare_task = create_task([]() {});

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
                = this->anchor_->info_
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
                //                    = this->anchor_->info_
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
            return make_shared_pages_consistent_impl(&shared_view_->access());
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
