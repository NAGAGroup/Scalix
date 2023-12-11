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

#include <scalix/defines.hpp>
#include <scalix/detail/page_handle.hpp>
#include <scalix/pointers.hpp>

namespace sclx::detail {

template<page_size_t PageSize = default_page_size>
class page_table_interface {
  public:
    using page_handle
        = sclx::detail::page_handle<page_handle_type::weak, PageSize>;
    using iterator = typename std::vector<page_handle>::iterator;

    virtual auto map_page(page_handle page) -> sclx::event                  = 0;
    virtual auto device_id() -> std::variant<device_id_t, sclx::mpi_device> = 0;
    virtual auto unmap_invalid_pages() -> sclx::event                       = 0;
    virtual auto begin() -> iterator                                        = 0;
    virtual auto end() -> iterator                                          = 0;

    virtual auto make_table_host_accessible() -> sclx::event   = 0;
    virtual auto make_table_device_accessible() -> sclx::event = 0;

    virtual ~page_table_interface() = default;
};

}  // namespace sclx::detail
