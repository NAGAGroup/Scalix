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

#include "shape.cuh"

namespace sclx {

using index_t = size_t;

template<uint Rank = 1>
class md_index_t : public shape_like_t<Rank> {
  public:
    __host__ __device__ constexpr md_index_t() : shape_like_t<Rank>() {}

    __host__ __device__ constexpr md_index_t(std::initializer_list<size_t> list)
        : shape_like_t<Rank>(list) {}

    __host__ __device__ constexpr md_index_t(const shape_like_t<Rank>& shape)
        : shape_like_t<Rank>(shape) {}

    __host__ __device__ index_t as_linear(const shape_t<Rank>& shape) const {
        index_t linear_index = 0;
        size_t stride        = 1;
        if constexpr (Rank > 1) {
            for (uint i = 0; i < Rank - 1; ++i) {
                linear_index += (*this)[i] * stride;
                stride *= shape[i];
            }
        }
        linear_index += (*this)[Rank - 1] * stride;
        return linear_index;
    }

    __host__ __device__ md_index_t operator+(const md_index_t& other) const {
        md_index_t result;
        for (uint i = 0; i < Rank; ++i) {
            result[i] = (*this)[i] + other[i];
        }
        return result;
    }

    __host__ __device__ md_index_t operator-(const md_index_t& other) const {
        md_index_t result;
        for (uint i = 0; i < Rank; ++i) {
            result[i] = (*this)[i] - other[i];
        }
        return result;
    }

    __host__ __device__ md_index_t& operator+=(const md_index_t& other) {
        for (uint i = 0; i < Rank; ++i) {
            (*this)[i] += other[i];
        }
        return *this;
    }

    __host__ __device__ md_index_t& operator-=(const md_index_t& other) {
        for (uint i = 0; i < Rank; ++i) {
            (*this)[i] -= other[i];
        }
        return *this;
    }

    template<uint OtherRank = Rank>
    static __host__ __device__ md_index_t<OtherRank>
    create_from_linear(index_t linear_index, const shape_t<OtherRank>& shape) {
        md_index_t<OtherRank> md_index;
        if constexpr (OtherRank > 1) {
            for (uint i = 0; i < OtherRank - 1; ++i) {
                md_index[i] = linear_index % shape[i];
                linear_index /= shape[i];
            }
        }
        md_index[OtherRank - 1] = linear_index;
        return md_index;
    }
};

}  // namespace sclx