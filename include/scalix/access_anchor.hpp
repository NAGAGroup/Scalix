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

auto future_to_event(sycl::queue queue, std::shared_future<void> fut)
    -> sycl::event {
    sycl::buffer<int, 1> buffer{1};
    std::promise<void> buffer_capture_promise;
    auto buffer_capture_future = buffer_capture_promise.get_future();
    std::thread t{[buffer_capture_promise = std::move(buffer_capture_promise),
                   buffer,
                   fut]() mutable {
        auto acsr = buffer.get_access<sycl::access_mode::discard_write>();
        buffer_capture_promise.set_value();
        acsr[0] = 1;
        fut.get();
    }};
    t.detach();

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

struct access_strategy_interface {
    enum access_locale : std::uint8_t { host, device };
    using access_marker     = signed char;
    using access_marker_ptr = access_marker*;

    virtual auto
    get_accessor_ready_tasks() -> const std::vector<generic_task>& = 0;

    virtual auto
    get_post_command_tasks() -> const std::vector<generic_task>& = 0;

    virtual auto get_anchor() const -> concurrent_guard<access_anchor> = 0;

    virtual void init_anchor_for_device(
        page_count_t page_count,
        const sycl::queue& device_queue
    ) = 0;

    [[nodiscard]] virtual auto
    get_page_ptrs(const sycl::device& device, access_locale locale) const
        -> page_ptr_t* = 0;

    [[nodiscard]] virtual auto
    get_device_page_access_markers(const sycl::device& device
    ) const -> access_marker_ptr* = 0;

    virtual void execute_strategy(
        sycl::queue device_queue,
        sycl::range<1> global_range,
        sycl::range<1> local_range,
        sycl::id<1> range_offset
    ) = 0;

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
    virtual auto allocate_page(sycl::queue queue
    ) const -> shared_ptr<detail::page_data_interface>                      = 0;
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

    auto allocate_page(sycl::queue queue
    ) const -> shared_ptr<detail::page_data_interface> override {
        auto& device_queue = queue;
        auto page_ptr      = make_shared<std::byte[]>(
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
                device_page_access_markers_;
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

    [[nodiscard]] auto get_device_page_access_markers(const sycl::device& device
    ) const -> access_strategy_interface::access_marker_ptr* {
        return info_->device_anchors_[device].device_page_access_markers_.get();
    }
};

struct access_strategy_common : access_strategy_interface {
    concurrent_guard<access_anchor> anchor_;

    auto get_anchor() const -> concurrent_guard<access_anchor> override {
        return anchor_;
    }

    void init_anchor_for_device(
        page_count_t page_count,
        const sycl::queue& device_queue
    ) override {
        auto anchor = anchor_.get_view<access_mode::write>(
            std::source_location::current()
        );
        if (anchor->info_ == nullptr) {
            anchor->info_ = std::make_unique<access_anchor::anchor_info>();
        }
        auto& device_anchor
            = anchor->info_->device_anchors_[device_queue.get_device()];
        device_anchor.pages_.resize(page_count);
        device_anchor.host_page_ptrs_.resize(page_count, nullptr);
        device_anchor.device_page_ptrs_ = std::move(make_unique<page_ptr_t[]>(
            device_queue,
            usm::alloc::device,
            page_count
        ));
        device_anchor.host_page_access_markers_.resize(page_count);
        device_anchor.device_page_access_markers_
            = std::move(make_unique<access_marker_ptr[]>(
                device_queue,
                usm::alloc::device,
                page_count
            ));
    }

    [[nodiscard]] auto
    get_page_ptrs(const sycl::device& device, access_locale locale) const
        -> page_ptr_t* override {
        auto anchor = anchor_.get_view<access_mode::read>(
            std::source_location::current()
        );
        return anchor->get_page_ptrs(device, locale);
    }

    [[nodiscard]] auto get_device_page_access_markers(const sycl::device& device
    ) const -> access_marker_ptr* override {
        auto anchor = anchor_.get_view<access_mode::read>(
            std::source_location::current()
        );
        return anchor->get_device_page_access_markers(device);
    }

    auto get_primary_queue() const -> sycl::queue {
        auto anchor = anchor_.get_view<access_mode::read>(
            std::source_location::current()
        );
        return sycl::queue{anchor->info_->device_anchors_.begin()->first};
    }

    auto get_primary_device() const -> const sycl::device& {
        auto anchor = anchor_.get_view<access_mode::read>(
            std::source_location::current()
        );
        return anchor->info_->device_anchors_.begin()->first;
    }

    sycl::event
    make_shared_pages_consistent_impl(buffer_helper_interface* buffer_helper
    ) const {
        //        auto primary_queue         = get_primary_queue();
        //        const auto& primary_device = get_primary_device();
        //        auto elements_per_page     =
        //        buffer_helper->get_elements_per_page(); auto element_size =
        //        buffer_helper->get_element_size();
        //
        //        std::vector<sycl::event> events;
        //
        //        // for the pages that are shared between devices, copy the
        //        data
        //        // to page data on the primary device
        //        using page_mapping_t
        //            = std::pair<sycl::device,
        //            shared_ptr<detail::page_data_interface>>;
        //        std::vector<std::vector<page_mapping_t>>
        //        mapped_peer_page_data(
        //            anchor_->info_->device_anchors_.begin()->second.pages_.size()
        //        );
        //        for (page_index_t page_idx = 0; page_idx <
        //        mapped_peer_page_data.size();
        //             ++page_idx) {
        //            std::vector<sycl::event> resident_events;
        //            for (auto& [device, device_anchor] :
        //                 anchor_->info_->device_anchors_) {
        //                auto& markers
        //                    =
        //                    device_anchor.host_page_access_markers_[page_idx];
        //                if (markers == nullptr && &device != &primary_device)
        //                {
        //                    continue;
        //                }
        //
        //                if (markers == nullptr && &device == &primary_device)
        //                {
        //                    markers = std::move(make_unique<access_marker[]>(
        //                        primary_queue,
        //                        usm::alloc::device,
        //                        elements_per_page
        //                    ));
        //                    primary_queue
        //                        .memset(
        //                            markers.get(),
        //                            -1,
        //                            buffer_helper->get_elements_per_page()
        //                        )
        //                        .wait_and_throw();
        //                }
        //                auto& peer_data = mapped_peer_page_data[page_idx];
        //                peer_data.emplace_back(
        //                    device,
        //                    buffer_helper->allocate_page(primary_device)
        //                );
        //                auto event = device_anchor.pages_[page_idx]->copy_to(
        //                    peer_data.back().second
        //                );
        //                resident_events.push_back(event);
        //            }
        //
        //            std::vector<page_mapping_t>& peer_data_list
        //                = mapped_peer_page_data[page_idx];
        //            const auto number_of_peers = peer_data_list.size();
        //            auto peer_pages            = make_unique<page_ptr_t[]>(
        //                primary_queue,
        //                usm::alloc::device,
        //                number_of_peers
        //            );
        //            auto peer_markers = make_unique<access_marker_ptr[]>(
        //                primary_queue,
        //                usm::alloc::device,
        //                number_of_peers
        //            );
        //
        //            std::vector<sycl::event> peer_access_info_events;
        //            for (int peer_idx = 1; peer_idx < peer_data_list.size();
        //                 ++peer_idx) {
        //                auto page_address
        //                    = peer_data_list[peer_idx].second->page_address();
        //                auto event = primary_queue.memcpy(
        //                    peer_pages.get() + peer_idx,
        //                    &page_address,
        //                    sizeof(page_ptr_t),
        //                    resident_events
        //                );
        //                event.wait_and_throw();
        //                peer_access_info_events.push_back(event);
        //                auto markers_address
        //                    = anchor->info_
        //                          ->device_anchors_[peer_data_list[peer_idx].first]
        //                          .host_page_access_markers_[page_idx]
        //                          .get();
        //                event = primary_queue.memcpy(
        //                    peer_markers.get() + peer_idx,
        //                    &markers_address,
        //                    sizeof(access_marker_ptr),
        //                    resident_events
        //                );
        //                event.wait_and_throw();
        //                peer_access_info_events.push_back(event);
        //            }
        //
        //            // update primary page with data from peers
        //            primary_queue
        //                .submit([&](sycl::handler& cgh) {
        //                    cgh.depends_on(peer_access_info_events);
        //
        //                    auto raw_peer_pages   = peer_pages.get();
        //                    auto raw_peer_markers = peer_markers.get();
        //                    cgh.parallel_for(
        //                        sycl::range{elements_per_page},
        //                        [raw_peer_pages,
        //                         raw_peer_markers,
        //                         number_of_peers,
        //                         elements_per_page,
        //                         element_size](sycl::id<> idx) {
        //                            auto primary_page_ptr = *raw_peer_pages;
        //                            auto primary_markers  = *raw_peer_markers;
        //                            for (int peer_idx = 1; peer_idx <
        //                            number_of_peers;
        //                                 ++peer_idx) {
        //                                auto peer_page_ptr
        //                                    = *(raw_peer_pages + peer_idx);
        //                                auto p_markers = *(raw_peer_markers +
        //                                peer_idx); if (primary_page_ptr ==
        //                                nullptr
        //                                    || peer_page_ptr == nullptr) {
        //                                    continue;
        //                                }
        //                                if (primary_page_ptr == peer_page_ptr)
        //                                {
        //                                    continue;
        //                                }
        //                                for (int elem = 0; elem <
        //                                elements_per_page;
        //                                     ++elem) {
        //                                    if (*(primary_markers + elem) !=
        //                                    -1
        //                                        || *(p_markers + elem) == 0) {
        //                                        continue;
        //                                    }
        //                                    std::memcpy(
        //                                        primary_page_ptr + elem *
        //                                        element_size, peer_page_ptr +
        //                                        elem * element_size,
        //                                        element_size
        //                                    );
        //                                }
        //                            }
        //                        }
        //                    );
        //                })
        //                .wait_and_throw();
        //
        //            // copy the updated primary page back to the peers
        //            std::vector<sycl::event> primary_to_peer_events;
        //            auto anchor = anchor_.get_view<access_mode::read>(
        //                std::source_location::current()
        //            );
        //            for (int peer_idx = 1; peer_idx < number_of_peers;
        //            ++peer_idx) {
        //                page_mapping_t& peer_data_pair =
        //                peer_data_list[peer_idx]; auto& peer_device_anchor
        //                    =
        //                    anchor->info_->device_anchors_[peer_data_pair.first];
        //                auto source_page      = peer_data_list.front().second;
        //                auto destination_page =
        //                peer_device_anchor.pages_[page_idx]; auto event =
        //                source_page->copy_to(destination_page);
        //                primary_to_peer_events.push_back(event);
        //            }
        //
        //            std::transform(
        //                primary_to_peer_events.begin(),
        //                primary_to_peer_events.end(),
        //                std::back_inserter(events),
        //                [](const sycl::event& event) { return event; }
        //            );
        //        }
        //
        //        auto combined_event = primary_queue.submit([&](sycl::handler&
        //        cgh) {
        //            cgh.depends_on(events);
        //            cgh.single_task([] {});
        //        });
        //        return combined_event;
        return {};
    }
};

struct handler {
    template<int RangeDimensions>
    struct parallel_for_config {
        std::vector<sycl::range<RangeDimensions>> local_ranges_;
        std::vector<sycl::id<RangeDimensions>> range_offsets_;
    };

    struct global_metadata {
        std::vector<double> weights_;
        std::unordered_map<void*, std::unique_ptr<access_strategy_interface>>
            strategies_;
        std::shared_ptr<void> command_config_;
    };

    struct metadata {
        sycl::queue device_queue_;
        size_t device_idx_;
        sycl::handler* device_handler_;
        global_metadata* global_metadata_;
        bool is_first_pass_{true};
        std::unordered_map<void*, std::shared_ptr<void>> type_erased_accessor_;
    };

    template<int RangeDimensions>
    auto get_parallel_for_config(const sycl::range<RangeDimensions>& range
    ) const -> const parallel_for_config<RangeDimensions>& {
        auto& global_metadata = *metadata_->global_metadata_;
        if (global_metadata.command_config_ != nullptr) {
            return *static_cast<parallel_for_config<RangeDimensions>*>(
                global_metadata.command_config_.get()
            );
        }

        global_metadata.command_config_
            = std::make_shared<parallel_for_config<RangeDimensions>>();
        auto config
            = std::static_pointer_cast<parallel_for_config<RangeDimensions>>(
                global_metadata.command_config_
            );
        auto range_dim0            = range[0];
        auto offset_dim0           = decltype(range_dim0){0};
        auto local_ranges_dim0_sum = decltype(range_dim0){0};
        for (double weight : global_metadata.weights_) {
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
        auto& meta          = *metadata_;
        auto& launch_config = get_parallel_for_config(range);
        auto local_range    = launch_config.local_ranges_[meta.device_idx_];
        auto range_offset   = launch_config.range_offsets_[meta.device_idx_];

        if (meta.is_first_pass_) {
            for (auto& strategy : meta.global_metadata_->strategies_) {
                strategy.second->execute_strategy(
                    meta.device_queue_,
                    range,
                    local_range,
                    range_offset
                );
            }
            return;
        }

        meta.device_handler_->parallel_for(
            local_range,
            [=](sycl::id<RangeDimensions> idx) { kernel(idx + range_offset); }
        );
    }

    template<
        access_mode AccessMode,
        class T,
        uint Dimensions,
        class AccessStrategy>
    auto assign_strategy(
        concurrent_guard<buffer_helper_base<T, Dimensions>> buffer_guard,
        const AccessStrategy& strategy
    ) -> accessor<T, Dimensions, AccessMode> {
        if (!metadata_->is_first_pass_) {
            auto acsr_ptr = metadata_->type_erased_accessor_
                                .at(&buffer_guard.unsafe_access())
                                .get();
            return *static_cast<accessor<T, Dimensions, AccessMode>*>(acsr_ptr);
        }
        auto& global_metadata       = *metadata_->global_metadata_;
        auto unprotected_buffer_ptr = &buffer_guard.unsafe_access();
        auto buffer_view = buffer_guard.template get_view<access_mode::write>(
            std::source_location::current()
        );
        if (global_metadata.strategies_.count(unprotected_buffer_ptr) == 0) {
            using strategy_type = std::unique_ptr<access_strategy_interface>;
            strategy_type strategy_impl
                = strategy.template get<AccessMode>(buffer_guard);
            strategy_impl->init_anchor_for_device(
                buffer_view->get_number_of_pages(),
                metadata_->device_queue_
            );
            global_metadata.strategies_[unprotected_buffer_ptr]
                = std::move(strategy_impl);
        }
        auto& strategy_impl
            = global_metadata.strategies_.at(unprotected_buffer_ptr);
        auto page_ptrs = strategy_impl->get_page_ptrs(
            device(),
            access_strategy_interface::access_locale::device
        );
        auto access_markers
            = strategy_impl->get_device_page_access_markers(device());
        using accessor_type = accessor<T, Dimensions, AccessMode>;
        auto shared_accessor_ptr
            = std::make_shared<accessor_type>(accessor_type{
                page_ptrs,
                buffer_view->range(),
                buffer_view->get_page_size(),
                access_markers
            });
        metadata_->type_erased_accessor_[unprotected_buffer_ptr]
            = shared_accessor_ptr;

        return *shared_accessor_ptr;
    }

    [[nodiscard]] auto device() const -> sycl::device {
        return metadata_->device_queue_.get_device();
    }

    std::unique_ptr<metadata> metadata_;
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
            for (auto& event : events) {
                event.wait_and_throw();
            }
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
            auto event
                = page->device_queue().submit([&events](sycl::handler& cgh) {
                      cgh.depends_on(events);
                      cgh.single_task([] {});
                  });
            event.wait_and_throw();
            return event;
        }

        [[nodiscard]] auto
        get_number_of_pages() const -> page_count_t override {
            return this->pages_.size();
        }
    };
    concurrent_guard<buffer_helper_base<T, Dimensions>> impl_{
        std::shared_ptr<impl<4096>>{}
    };

    template<access_mode AccessMode = access_mode::read_write>
    auto get_access() const
        -> host_accessor<
            std::conditional_t<AccessMode == access_mode::read, const T, T>,
            Dimensions> {
        auto view = impl_.template get_view<AccessMode>(
            std::source_location::current()
        );
        auto acsr_generic = view->get_host_access();
        using acsr_type   = host_accessor<
              std::conditional_t<AccessMode == access_mode::read, const T, T>,
              Dimensions>;
        acsr_type acsr;
        acsr.data_          = std::move(acsr_generic.data_);
        acsr.buffer_handle_ = std::make_shared<decltype(view)>(std::move(view));

        return acsr;
    }

    explicit buffer(range<Dimensions> range)
        : impl_{std::make_shared<impl<4096>>()} {
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
        auto acsr
            = cgh.assign_strategy<AccessMode, T, Dimensions, AccessStrategy>(
                impl_,
                strategy
            );

        return acsr;
    }
};

struct queue {
    struct command_metadata {
        std::vector<handler> command_handlers;
        handler::global_metadata global_metadata;
    };

    template<class Submission>
    auto submit(Submission&& submission) -> sycl::event {
        auto cmd_metadata         = std::make_unique<command_metadata>();
        auto global_metadata      = &cmd_metadata->global_metadata;
        global_metadata->weights_ = device_weights_;

        cmd_metadata->command_handlers.resize(device_queues_.size());
        for (int dev_idx = 0; dev_idx < device_queues_.size(); ++dev_idx) {
            auto& dqueue      = device_queues_[dev_idx];
            auto& handler     = cmd_metadata->command_handlers[dev_idx];
            handler.metadata_ = std::make_unique<handler::metadata>();
            handler.metadata_->device_queue_    = dqueue;
            handler.metadata_->device_idx_      = dev_idx;
            handler.metadata_->global_metadata_ = global_metadata;
            submission(handler);
        }

        auto command_ready_task = create_task([]() {});
        for (auto& [unused, strategy] : global_metadata->strategies_) {
            for (auto& task : strategy->get_accessor_ready_tasks()) {
                task.add_dependent_task(command_ready_task);
                task.launch();
            }
        }
        command_ready_task.launch();
        auto command_ready_event = future_to_event(
            device_queues_.front(),
            command_ready_task.get_future()
        );

        std::vector<sycl::event> command_events;
        for (auto& handler : cmd_metadata->command_handlers) {
            handler.metadata_->is_first_pass_ = false;
            auto& dqueue = handler.metadata_->device_queue_;
            auto event   = dqueue.submit([&](sycl::handler& cgh) {
                handler.metadata_->device_handler_ = &cgh;
                cgh.depends_on(command_ready_event);
                submission(handler);
            });
            command_events.push_back(event);
        }

        auto events_wait_task = create_task([command_events]() mutable {
            for (auto& event : command_events) {
                event.wait_and_throw();
            }
            std::cout << "Command events finished\n";
        });

        auto cleanup_task
            = create_task([cmd_metadata = std::move(cmd_metadata)]() mutable {
                  cmd_metadata.reset();
                  std::cout << "Cleanup task finished\n";
              });

        for (auto& [unused, strategy] : global_metadata->strategies_) {
            for (auto& task : strategy->get_post_command_tasks()) {
                events_wait_task.add_dependent_task(task);
                task.add_dependent_task(cleanup_task);
                task.launch();
            }
        }
        events_wait_task.launch();
        cleanup_task.launch();

        auto fut           = cleanup_task.get_future().share();
        auto cleanup_event = future_to_event(device_queues_.front(), fut);
        fut.get();

        return cleanup_event;

        //        std::vector<handler> command_handlers(device_weights_.size());
        //        std::vector<sycl::event> command_events;
        //        auto command_topology_guard
        //            = concurrent_guard<handler::metadata::global_metadata>();
        //        auto& command_topology    =
        //        command_topology_guard.unsafe_access();
        //        command_topology.weights_ = device_weights_;
        //        for (auto& dqueue : device_queues_) {
        //            auto idx          = std::distance(&device_queues_.front(),
        //            &dqueue); auto& handler     = command_handlers[idx];
        //            handler.metadata_ = std::make_shared<handler::metadata>();
        //            handler.metadata_->device_queue_ = dqueue;
        //            handler.metadata_->device_       = dqueue.get_device();
        //            handler.metadata_->device_idx_
        //                = std::distance(&device_queues_.front(), &dqueue);
        //            handler.metadata_->global_metadata_ =
        //            command_topology_guard; submission(handler);
        //            command_events.push_back(handler.get_command_event());
        //        }
        //
        //        auto global_command_task = create_task(
        //            [](handler primary_handler,
        //               sycl::queue primary_queue,
        //               std::vector<sycl::event> cmd_events) {
        //                for (auto& event : cmd_events) {
        //                    event.wait_and_throw();
        //                }
        //                std::vector<sycl::event> finalization_events;
        //                auto global_metadata
        //                    =
        //                    primary_handler.metadata_->global_metadata_.unsafe_access(
        //                    );
        //                for (auto& [unused, strategy] :
        //                global_metadata.strategies_) {
        //                    finalization_events.push_back(
        //                        strategy->make_shared_pages_consistent()
        //                    );
        //                }
        //
        //                return finalization_events;
        //            },
        //            command_handlers.front(),
        //            device_queues_.front(),
        //            command_events
        //        );
        //        auto fut = global_command_task.get_future();
        //        global_command_task.launch();
        //        auto finalization_events = fut.get();
        //
        //        auto global_command_event = device_queues_.front().submit(
        //            [&finalization_events](sycl::handler& cgh) {
        //                cgh.depends_on(finalization_events);
        //                cgh.single_task([] {});
        //            }
        //        );

        //        return global_command_event;
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

        std::vector<generic_task> accessor_ready_tasks_;
        std::vector<generic_task> post_command_tasks_;

        auto get_accessor_ready_tasks()
            -> const std::vector<generic_task>& override {
            return accessor_ready_tasks_;
        }

        auto
        get_post_command_tasks() -> const std::vector<generic_task>& override {
            return post_command_tasks_;
        }

        void execute_strategy(
            sycl::queue device_queue,
            sycl::range<1> global_range,
            sycl::range<1> local_range,
            sycl::id<1> range_offset
        ) override {
            std::promise<void> lock_task_started;
            auto lock_task_started_future = lock_task_started.get_future();
            std::promise<void> buffer_locked_promise;
            auto buffer_locked_future = buffer_locked_promise.get_future();
            auto lock_task            = create_task([&lock_task_started,
                                          buffer_locked_promise
                                          = std::move(buffer_locked_promise),
                                          this]() mutable {
                auto is_locked = is_locked_.exchange(true);
                if (!is_locked) {
                    lock_task_started.set_value();
                    locked_buffer_view_ = std::make_unique<buffer_view_t>(
                        buffer_helper_.template get_view<AccessMode>()
                    );
                    is_locking_.exchange(false);
                } else {
                    lock_task_started.set_value();
                }
                while (is_locking_.load()) {
                    std::this_thread::yield();
                }
                auto buffer_view = locked_buffer_view_;
                buffer_locked_promise.set_value();
                while (buffer_view.use_count() > 1) {}
                std::cout << "Made it here!\n";
            });
            lock_task.launch();
            lock_task_started_future.wait();
            auto prepare_task = create_task([=,
                                             buffer_locked_future
                                             = std::move(buffer_locked_future),
                                             this]() mutable {
                buffer_locked_future.get();
                auto& buffer_ptr = *locked_buffer_view_.get();
                auto anchor      = anchor_.get_view<access_mode::read>(
                    std::source_location::current()
                );
                auto& device_anchor
                    = anchor->info_->device_anchors_[device_queue.get_device()];
                auto& pages = device_anchor.pages_;
                std::transform(
                    pages.begin(),
                    pages.end(),
                    pages.begin(),
                    [&](auto& page) {
                        return buffer_ptr->allocate_page(device_queue);
                    }
                );
                buffer_ptr->make_pages_valid(pages);
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
                        device_anchor.host_page_ptrs_.size()
                            * sizeof(page_ptr_t)
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

                    std::for_each(
                        memset_events.begin(),
                        memset_events.end(),
                        [](auto& event) { event.wait_and_throw(); }
                    );
                }

                device_queue
                    .memcpy(
                        device_anchor.device_page_access_markers_.get(),
                        device_anchor.host_page_access_markers_.data(),
                        device_anchor.host_page_access_markers_.size()
                            * sizeof(access_marker_ptr)
                    )
                    .wait_and_throw();
            });

            this->accessor_ready_tasks_.push_back(prepare_task);

            if constexpr (AccessMode == access_mode::read) {
                return;
            }

            auto finalization_task = create_task([=, this]() mutable {
                auto anchor = anchor_.get_view<access_mode::read>(
                    std::source_location::current()
                );
                auto& buffer_ptr = *locked_buffer_view_.get();
                auto& device_anchor
                    = anchor->info_->device_anchors_[device_queue.get_device()];
                std::vector<sycl::event> update_events;
                for (auto& page : device_anchor.pages_) {
                    auto page_index = static_cast<page_index_t>(
                        std::distance(&device_anchor.pages_.front(), &page)
                    );
                    auto event
                        = buffer_ptr->update_page_data_for(page_index, page);
                    update_events.push_back(event);
                }
                for (auto& event : update_events) {
                    event.wait_and_throw();
                }
            });
            prepare_task.add_dependent_task(finalization_task);
            this->post_command_tasks_.push_back(finalization_task);
        }

        auto make_shared_pages_consistent() -> sycl::event override {
            if constexpr (AccessMode == access_mode::read) {
                return {};
            }
            return make_shared_pages_consistent_impl(
                &locked_buffer_view_->access()
            );
        }

        ~impl() { std::cout << "destroyed\n"; }

        impl(concurrent_guard<buffer_helper_interface> buffer_helper)
            : buffer_helper_{buffer_helper} {}

        impl(const impl&)                    = delete;
        impl(impl&&)                         = default;
        auto operator=(const impl&) -> impl& = delete;
        auto operator=(impl&&) -> impl&      = default;

        std::atomic<bool> is_locking_{true};
        std::atomic<bool> is_locked_{false};
        std::shared_ptr<buffer_view_t> locked_buffer_view_;
        concurrent_guard<buffer_helper_interface> buffer_helper_{nullptr};
    };

    template<access_mode AccessMode, class T, int Dimensions>
    auto get(concurrent_guard<buffer_helper_base<T, Dimensions>> buffer
    ) const -> std::unique_ptr<impl<AccessMode>> {
        return std::unique_ptr<impl<AccessMode>>{new impl<AccessMode>{buffer}};
    }
};

}  // namespace sclx
