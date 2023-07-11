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

#include "constexpr_assign_array.cuh"
#include "throw_exception.hpp"
#include <initializer_list>

namespace sclx {

namespace detail {

template<uint N, class T>
__host__ __device__ constexpr bool arrays_equal(const T* a, const T* b) {
    if (a[0] != b[0]) {
        return false;
    }
    if constexpr (N > 1) {
        return arrays_equal<N - 1>(a + 1, b + 1);
    }
    return true;
}

}  // namespace detail

template<uint Rank>
class shape_like_t {
  public:
    constexpr shape_like_t() = default;

    __host__
        __device__ constexpr shape_like_t(std::initializer_list<size_t> list) {
        if (list.size() != Rank) {
#ifndef __CUDA_ARCH__
            throw_exception<std::invalid_argument>(
                "Shape initializer list size does not match rank",
                "sclx::shape_like_t::"
            );
#endif
        }
        constexpr_assign_array<Rank>(shape_, list.begin());
    }

    __host__ __device__ constexpr shape_like_t(const shape_like_t& other) {
        constexpr_assign_array<Rank>(shape_, other.shape_);
    }

    __host__ __device__ constexpr shape_like_t(const size_t (&shape)[Rank]) {
        constexpr_assign_array<Rank>(shape_, shape);
    }

    __host__ __device__ const size_t& operator[](size_t i) const {
        return shape_[i];
    }

    __host__ __device__ size_t& operator[](size_t i) { return shape_[i]; }

    [[nodiscard]] __host__ __device__ constexpr uint rank() const {
        return Rank;
    }

    __host__ __device__ bool operator==(const shape_like_t& other) const {
        return detail::arrays_equal<Rank>(shape_, other.shape_);
    }

    __host__ __device__ bool operator!=(const shape_like_t& other) const {
        return !detail::arrays_equal<Rank>(shape_, other.shape_);
    }

    __host__ friend std::ostream&
    operator<<(std::ostream& os, const shape_like_t& shape) {
        os << "(";
        for (uint i = 0; i < Rank; ++i) {
            os << shape[i];
            if (i != Rank - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }

  protected:
    size_t shape_[Rank]{0};
};

template<uint Rank>
class shape_t : public shape_like_t<Rank> {
  public:
    using shape_like_t<Rank>::shape_like_t;

    __host__ __device__ size_t elements() const {
        size_t elements = 1;
        for (uint i = 0; i < Rank; ++i) {
            elements *= this->shape_[i];
        }
        return elements;
    }
};

}  // namespace sclx