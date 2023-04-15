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

#include "array.cuh"

namespace sclx {

__host__ bool is_same_device_split(
    const std::vector<std::tuple<int, size_t, size_t>>& lhs,
    const std::vector<std::tuple<int, size_t, size_t>>& rhs
) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::get<0>(lhs[i]) != std::get<0>(rhs[i])) {
            return false;
        }
        if (std::get<1>(lhs[i]) != std::get<1>(rhs[i])) {
            return false;
        }
        if (std::get<2>(lhs[i]) != std::get<2>(rhs[i])) {
            return false;
        }
    }
    return true;
}

SCALIX_INSTANTIATE_ARRAY_LIKE(class, array, float, )
SCALIX_INSTANTIATE_ARRAY_LIKE(class, array, double, )
SCALIX_INSTANTIATE_ARRAY_LIKE(class, array, int, )
SCALIX_INSTANTIATE_ARRAY_LIKE(class, array, uint, )
SCALIX_INSTANTIATE_ARRAY_LIKE(class, array, size_t, )

template<class T, uint Rank>
using get_device_split_info_t = decltype(get_device_split_info<T, Rank>);
SCALIX_INSTANTIATE_ARRAY_LIKE(
    ,
    __host__ get_device_split_info_t,
    float,
    get_device_split_info
)
SCALIX_INSTANTIATE_ARRAY_LIKE(
    ,
    __host__ get_device_split_info_t,
    double,
    get_device_split_info
)
SCALIX_INSTANTIATE_ARRAY_LIKE(
    ,
    __host__ get_device_split_info_t,
    int,
    get_device_split_info
)
SCALIX_INSTANTIATE_ARRAY_LIKE(
    ,
    __host__ get_device_split_info_t,
    uint,
    get_device_split_info
)
SCALIX_INSTANTIATE_ARRAY_LIKE(
    ,
    __host__ get_device_split_info_t,
    size_t,
    get_device_split_info
)

}
