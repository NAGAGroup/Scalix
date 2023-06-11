
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

#pragma once
#include "../array.cuh"
#include "../execute_kernel.cuh"
#include "../throw_exception.hpp"
#include "functional.cuh"

namespace sclx::algorithm {

REGISTER_SCALIX_KERNEL_TAG(transform_kernel);

template<class R, class T, uint Rank, class U, class BinaryOp>
void transform(
    sclx::array<R, Rank>& result,
    const sclx::array<T, Rank>& arr,
    const U& scalar,
    BinaryOp&& op
) {
    if (result.shape() != arr.shape()) {
        throw_exception<std::invalid_argument>("input shapes must match"
                                               "sclx::algorithm::");
    }
    sclx::execute_kernel([&](const kernel_handler& handler) {
        handler.launch<transform_kernel>(
            md_range_t<Rank>(result.shape()),
            result,
            [=] __device__(const md_index_t<Rank>& idx, const auto& info) {
                const auto& thread = info.global_thread_linear_id();
                result[thread] = op(arr[thread], scalar);
            }
        );
    }).get();
}

}  // namespace sclx::algorithm
