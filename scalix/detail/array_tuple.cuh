
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

namespace sclx::detail {

template<class T>
__host__ __device__ bool is_last_dim_of_arrays_equal(T&& a) {
    static_assert(
        is_scalix_array_v<T>,
        "is_last_dim_of_arrays_equal must be called with scalix arrays"
    );
    return true;
}

template<class T1, class T2, class... Types>
__host__ __device__ bool
is_last_dim_of_arrays_equal(T1&& a1, T2&& a2, Types&&... arrays) {
    static_assert(
        is_scalix_arrays_v<T1, T2, Types...>,
        "is_last_dim_of_arrays_equal must be called with scalix arrays"
    );
    auto last_dim1 = a1.shape()[a1.rank() - 1];
    auto last_dim2 = a2.shape()[a2.rank() - 1];
    if (last_dim1 != last_dim2) {
        return false;
    }
    return is_last_dim_of_arrays_equal(a2, arrays...);
}

template<class F, uint N, class Tuple, class... Types>
__host__ __device__ auto apply(F&& f, Tuple t, Types&&... args) {
    if constexpr (N == 0) {
        return f(std::forward<Types>(args)...);
    } else {
        return apply<
            F,
            N - 1,
            Tuple,
            decltype(thrust::get<N - 1>(t)),
            Types...>(
            std::forward<F>(f),
            t,
            thrust::get<N - 1>(t),
            std::forward<Types>(args)...
        );
    }
}

}  // namespace sclx::detail
