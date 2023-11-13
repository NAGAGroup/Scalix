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
#include "../throw_exception.hpp"
#include "functional.cuh"

namespace sclx::algorithm {

template<class T, class U, uint Rank, class... Ts>
bool check_shapes(
    const sclx::array<T, Rank>& a,
    const sclx::array<U, Rank>& b,
    const sclx::array<Ts, Rank>&... args
) {
    if (a.shape() != b.shape()) {
        return false;
    }
    if constexpr (sizeof...(args) > 0) {
        return check_shapes(b, args...);
    }
    return true;
}

template<class R, class T, class U, uint Rank, class... Ts>
struct reduce_along_index {
    sclx::array<T, Rank> a_;
    sclx::array<U, Rank> b_;
    thrust::tuple<sclx::array<Ts, Rank>...> args_;
    constexpr static uint num_args = sizeof...(Ts);

    __host__ reduce_along_index(
        const sclx::array<T, Rank>& a,
        const sclx::array<U, Rank>& b,
        const sclx::array<Ts, Rank>&... args
    )
        : a_(a),
          b_(b),
          args_(args...) {}

    template<class BinaryOp, uint I>
    __host__ __device__ R
    compute_impl(const index_t& idx, BinaryOp&& op, const thrust::integral_constant<uint, I>&)
        const {
        if constexpr (I > 0) {
            return op(
                compute_impl(idx, op, thrust::integral_constant<uint, I - 1>()),
                thrust::get<num_args - I>(args_)[idx]
            );
        } else {
            return op(a_[idx], b_[idx]);
        }
    }

    template<class BinaryOp>
    __host__ __device__ R compute(const index_t& idx, BinaryOp&& op) const {
        return compute_impl(
            idx,
            op,
            thrust::integral_constant<uint, num_args>()
        );
    }
};

REGISTER_SCALIX_KERNEL_TAG(elementwise_reduce_kernel);

template<class BinaryOp, class R, class T, class U, uint Rank, class... Ts>
std::future<void> elementwise_reduce(
    BinaryOp&& op,
    sclx::array<R, Rank>& result,
    const sclx::array<T, Rank>& a,
    const sclx::array<U, Rank>& b,
    const sclx::array<Ts, Rank>&... args
) {
    if (!check_shapes(a, b, args...)) {
        throw_exception<std::invalid_argument>("input shapes must match"
                                               "sclx::algorithm::");
    }

    reduce_along_index<R, T, U, Rank, Ts...> functor(a, b, args...);

    return sclx::execute_kernel([=](const kernel_handler& handler) {
        handler.launch<elementwise_reduce_kernel>(
            md_range_t<Rank>(result.shape()),
            result,
            [=] __device__(const md_index_t<Rank>& idx, const auto& info) {
                const auto& thread = info.global_thread_linear_id();
                result[thread]     = functor.compute(thread, op);
            }
        );
    });
}

}  // namespace sclx::algorithm
