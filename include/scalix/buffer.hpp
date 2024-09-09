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
#include <scalix/generic_task.hpp>

namespace sclx {
template<class T, int Dimensions, access_mode AccessMode>
class accessor;

class access_anchor;

namespace detail {

class buffer_handle {
  public:
    buffer_handle()                                        = default;
    buffer_handle(const buffer_handle&)                    = default;
    auto operator=(const buffer_handle&) -> buffer_handle& = default;
    buffer_handle(buffer_handle&&)                         = default;
    auto operator=(buffer_handle&&) -> buffer_handle&      = default;

    virtual ~buffer_handle() = default;
};

template<class T, int Dimensions>
class buffer_interface : public buffer_handle {
  protected:
    struct accessor_metadata {};

    virtual auto
    create_partition_data(const sycl::queue& assoc_queue, access_anchor& anchor)
        -> std::vector<std::shared_ptr<nd_partition<T, Dimensions>>> = 0;

    virtual auto create_partition_data(
        const sycl::queue& assoc_queue,
        access_anchor& anchor,
        const std::vector<bool>& accessed_partitions
    ) -> std::vector<std::shared_ptr<nd_partition<T, Dimensions>>> = 0;

    virtual void
    populate_command_dependendency_for_read_access(generic_task& command)
        = 0;

    virtual void
    populate_command_dependendency_for_write_access(generic_task& command)
        = 0;

  private:
    template<access_mode AccessMode>
    auto
    create_accessor(const sycl::queue& assoc_queue, const access_anchor& anchor)
        -> accessor<T, Dimensions, AccessMode>;
};

}  // namespace detail

template<class T, int Dimensions>
class buffer {};

}  // namespace sclx
