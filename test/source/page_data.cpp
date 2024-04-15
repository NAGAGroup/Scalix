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
#include <catch2/catch_test_macros.hpp>
#include <scalix/detail/page_data.hpp>

TEST_CASE("page_data") {
    const auto data_ptr1
        = std::make_shared<sclx::byte[]>(sclx::default_page_size);
    const sclx::detail::page_data<sclx::default_page_size> page_data1{
        data_ptr1.get(),
        data_ptr1
    };
    using data_type = std::int32_t;
    constexpr sclx::page_size_t floats_per_page
        = sclx::default_page_size / sizeof(data_type);
    for (sclx::page_size_t i = 0; i < floats_per_page; ++i) {
        reinterpret_cast<data_type*>(data_ptr1.get())[i]
            = static_cast<data_type>(i);
    }

    const auto data_ptr2
        = std::make_shared<sclx::byte[]>(sclx::default_page_size);
    sclx::detail::page_data<sclx::default_page_size> page_data2{
        data_ptr2.get(),
        data_ptr2
    };
    page_data1.copy_to(page_data2);

    const auto data_ptr3
        = std::make_shared<sclx::byte[]>(sclx::default_page_size);
    const sclx::detail::page_data<sclx::default_page_size> page_data3{
        data_ptr3.get(),
        data_ptr3
    };
    page_data2.copy_to(data_ptr3.get());

    const auto page_data3_raw
        = reinterpret_cast<const data_type*>(page_data3.page_address());
    const auto page_data1_raw
        = reinterpret_cast<const data_type*>(page_data1.page_address());
    for (sclx::page_size_t i = 0; i < floats_per_page; ++i) {
        REQUIRE(page_data3_raw[i] == page_data1_raw[i]);
    }
}
