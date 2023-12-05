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
#include <sycl.hpp>

namespace sclx {

// constants
constexpr std::uint32_t default_page_size = 4096;

// types
using byte      = std::byte;
using write_bit_t = char;
using size_t    = std::size_t;
using index32_t = std::int32_t;
using index64_t = std::int64_t;
using uint64_t  = std::uint64_t;
using uint32_t  = std::uint32_t;

using page_size_t = std::uint32_t;
using page_ptr_t  = byte*;
using valid_bit_t = std::atomic<bool>;
// On any system with less than 8TiB of memory and a 4KB page size,
// a 32-bit page index is sufficient. However, if you are defining
// buffers with smaller page sizes or on a system with more than 16TiB
// of memory, you may need to define SCLX_PAGE_INDEX_TYPE to a 64-bit
// unsigned integer type.
#ifndef SCLX_USE_LONG_PAGE_INDEX
using page_index_t = std::int32_t;
using page_diff_t  = std::int32_t;
using page_count_t = std::uint32_t;
#else
using page_index_t = std::int64_t;
using page_diff_t  = std::int64_t;
using page_count_t = std::uint64_t;
#endif

constexpr page_index_t invalid_page = -1;

using sycl::half;

// exposed sycl namespaced items
using sycl::access_mode;
using sycl::device;
using sycl::event;
using sycl::id;

// additional sycl-like types
using device_id_t = int;
constexpr device_id_t host_device_id = 0;
constexpr device_id_t no_device = -1;
using rank_id_t   = int;
struct mpi_device {
    rank_id_t rank;
    device_id_t device_id;
};

namespace usm {
using sycl::usm::alloc;
}

namespace access {
using sycl::access::target;
}

namespace info = sycl::info;

}  // namespace sclx
