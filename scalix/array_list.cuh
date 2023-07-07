
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

namespace sclx {
template<class T, uint Rank, uint N>
class array_list {
  public:
    template<class T_ = const T>
    __host__ operator array_list<T_, Rank, N>() const {
        static_assert(std::is_same_v<T_, const T>, "Can only cast to const");
        std::vector<array<T_, Rank>> const_arrays(arrays_, arrays_ + N);
        return array_list<T_, Rank, N>(const_arrays);
    }

    __host__ array_list(std::initializer_list<array<T, Rank>> arrays) {
        if (arrays.size() != N) {
            throw_exception<std::invalid_argument>(
                "array_list must be initialized with exactly N arrays",
                "sclx::array_list::"
            );
        }
        std::copy(arrays.begin(), arrays.end(), arrays_);
    }

    __host__ explicit array_list(const std::vector<array<T, Rank>>& arrays) {
        if (arrays.size() != N) {
            throw_exception<std::invalid_argument>(
                "array_list must be initialized with exactly N arrays",
                "sclx::array_list::"
            );
        }
        std::copy(arrays.begin(), arrays.end(), arrays_);
    }

    __host__ explicit array_list(const array<T, Rank>* arrays) {
        std::copy(arrays, arrays + N, arrays_);
    }

    __host__ explicit array_list(const std::array<array<T, Rank>, N>& arrays) {
        std::copy(arrays.begin(), arrays.end(), arrays_);
    }

    __host__ __device__ array<T, Rank>& operator[](size_t i) {
        return arrays_[i];
    }

    __host__ __device__ const array<T, Rank>& operator[](size_t i) const {
        return arrays_[i];
    }

    __host__ __device__ array<T, Rank>* begin() { return arrays_; }

    __host__ __device__ array<T, Rank>* end() { return arrays_ + N; }

    __host__ __device__ const array<T, Rank>* begin() const { return arrays_; }

    __host__ __device__ const array<T, Rank>* end() const {
        return arrays_ + N;
    }

  private:
    array<T, Rank> arrays_[N];
};

template<class T, uint Rank>
class dynamic_array_list {
  public:
    dynamic_array_list() = default;

    dynamic_array_list(const dynamic_array_list&)                = default;
    dynamic_array_list(dynamic_array_list&&) noexcept            = default;
    dynamic_array_list& operator=(const dynamic_array_list&)     = default;
    dynamic_array_list& operator=(dynamic_array_list&&) noexcept = default;

    __host__ dynamic_array_list(const std::vector<array<T, Rank>>& arrays) {
        data_ = detail::allocate_cuda_usm<array<T, Rank>>(arrays.size());
        size_ = arrays.size();
        std::copy(arrays.begin(), arrays.end(), data_.get());
    }

    __host__
    dynamic_array_list(const std::initializer_list<array<T, Rank>>& arrays) {
        data_ = detail::allocate_cuda_usm<array<T, Rank>>(arrays.size());
        size_ = arrays.size();
        std::copy(arrays.begin(), arrays.end(), data_.get());
    }

    __host__ __device__ array<T, Rank>& operator[](size_t i) {
        return data_.get()[i];
    }

    __host__ __device__ const array<T, Rank>& operator[](size_t i) const {
        return data_.get()[i];
    }

    __host__ __device__ const size_t& size() const { return size_; }

    __host__ __device__ array<T, Rank>* begin() { return data_.get(); }

    __host__ __device__ array<T, Rank>* end() { return data_.get() + size_; }

    __host__ __device__ const array<T, Rank>* begin() const {
        return data_.get();
    }

    __host__ __device__ const array<T, Rank>* end() const {
        return data_.get() + size_;
    }

  private:
    detail::unified_ptr<T> data_{};  // pointer to data
    size_t size_{};                  // number of elements
};
}  // namespace sclx
