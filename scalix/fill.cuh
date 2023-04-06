
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

#include "array.cuh"
#include "execute_kernel.cuh"

namespace sclx {

template<class T, uint Rank>
void fill(const array<T, Rank>& arr, const T& value) {
    execute_kernel([&](kernel_handler& handler) {
        handler.launch(
            md_range_t<Rank>(arr.shape()),
            arr,
            [=] __device__(const md_index_t<Rank>& index, const auto&) {
                arr[index] = value;
            }
        );
    }).get();
}

template<class T, uint Rank>
__host__ array<T, Rank> zeros(const size_t (&shape)[Rank]) {
    array<T, Rank> arr{shape_t<Rank>(shape)};
    fill(arr, T(0));
    return arr;
}

template<class T, uint Rank>
__host__ array<T, Rank> zeros(const shape_t<Rank>& shape) {
    array<T, Rank> arr{shape};
    fill(arr, T(0));
    return arr;
}

template<class T, uint Rank>
__host__ array<T, Rank> ones(const size_t (&shape)[Rank]) {
    array<T, Rank> arr{shape_t<Rank>(shape)};
    fill(arr, T(1));
    return arr;
}

template<class T, uint Rank>
__host__ array<T, Rank> ones(const shape_t<Rank>& shape) {
    array<T, Rank> arr{shape};
    fill(arr, T(1));
    return arr;
}

}  // namespace sclx
