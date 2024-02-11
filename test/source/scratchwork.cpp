// BSD 3-Clause License
//
// Copyright (c) 2023-2024 Jack Myers
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

#include "scalix/detail/host_page_table.hpp"

#include <numeric>
#include <scalix/detail/device_allocation.hpp>
#include <scalix/detail/device_page_table.hpp>
#include <scalix/detail/host_allocation.hpp>

using value_type = float;
auto page_count
    = sclx::detail::required_pages_for_elements<value_type>(100'000'000);

template<template<
    sclx::detail::pagination_type,
    class,
    sclx::detail::reuse_pages,
    sclx::page_size_t>
         class AllocationType>
void test_allocations(
    sclx::detail::page_table_interface<sclx::default_page_size>& page_table
) {
    std::cout << "page_count: " << page_count << std::endl;

    auto device_id = std::get<sclx::device_id_t>(page_table.device_id());
    AllocationType<
        sclx::detail::pagination_type::contiguous,
        value_type,
        sclx::detail::reuse_pages::if_possible,
        sclx::default_page_size>
        allocation1(device_id, 0, page_count, {});

    std::vector<sclx::page_index_t> indices(page_count);
    std::iota(indices.begin(), indices.end(), 0);
    AllocationType<
        sclx::detail::pagination_type::contiguous,
        value_type,
        sclx::detail::reuse_pages::if_possible,
        sclx::default_page_size>
        allocation2(device_id, indices, {});

    auto page_handles1 = sclx::detail::access_anchor_interface::get_allocation<
                             sclx::default_page_size>(allocation1.anchor())
                             .pages();
    std::vector<sclx::event> events;
    std::transform(
        page_handles1.begin(),
        page_handles1.end(),
        std::back_inserter(events),
        [&](auto& page_handle) { return page_table.map_page(page_handle); }
    );

    auto page_handles2 = sclx::detail::access_anchor_interface::get_allocation<
                             sclx::default_page_size>(allocation2.anchor())
                             .pages();
    std::transform(
        page_handles2.begin(),
        page_handles2.end(),
        std::back_inserter(events),
        [&](auto& page_handle) { return page_table.map_page(page_handle); }
    );

    for (auto& event : events) {
        event.wait_and_throw();
    }

    AllocationType<
        sclx::detail::pagination_type::paginated,
        value_type,
        sclx::detail::reuse_pages::if_possible,
        sclx::default_page_size>
        allocation3(device_id, 0, page_count, page_table.pages());
    auto page_handles3 = sclx::detail::access_anchor_interface::get_allocation<
                             sclx::default_page_size>(allocation3.anchor())
                             .pages();

    events.clear();
    std::transform(
        page_handles3.begin(),
        page_handles3.end(),
        std::back_inserter(events),
        [&](auto& page_handle) { return page_table.map_page(page_handle); }
    );

    allocation1
        = std::move(allocation2);  // this will remove any allocation handles
    // from the factory, meaning we can't
    // reuse them anymore from the page table, as they are not valid anymore
    AllocationType<
        sclx::detail::pagination_type::paginated,
        value_type,
        sclx::detail::reuse_pages::if_possible,
        sclx::default_page_size>
        allocation4(device_id, indices, page_table.pages());

    for (auto& event : events) {
        event.wait_and_throw();
    }

    page_handles1 = sclx::detail::access_anchor_interface::get_allocation<
                        sclx::default_page_size>(allocation1.anchor())
                        .pages();
    events.clear();
    std::transform(
        page_handles1.begin(),
        page_handles1.end(),
        std::back_inserter(events),
        [&](auto& page_handle) { return page_table.map_page(page_handle); }
    );

    for (auto& event : events) {
        event.wait_and_throw();
    }
}

auto main() -> int {
    auto ptr = std::make_shared<float[]>(5);
    std::cout << "ptr: " << ptr[5] << std::endl;

    sclx::device device;
    // get a cuda device
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu()) {
            device = sclx::device(d);
            break;
        }
    }

    sclx::detail::device_page_table device_page_table{device, page_count};
    test_allocations<sclx::detail::device_allocation>(device_page_table);

    sclx::detail::host_page_table host_page_table{page_count};
    test_allocations<sclx::detail::host_allocation>(host_page_table);

    return 0;
}
