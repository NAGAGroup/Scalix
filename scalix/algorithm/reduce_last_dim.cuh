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
#include "../array.cuh"
#include "../execute_kernel.cuh"
#include "../fill.cuh"
#include "functional.cuh"
#include <mutex>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace sclx::algorithm {

template<class T>
class strided_iterator {
  public:
    __host__ __device__ strided_iterator(T* ptr, size_t stride)
        : ptr_(ptr),
          stride_(stride) {}

    __host__ __device__ strided_iterator& operator++() {
        ptr_ += stride_;
        return *this;
    }

    __host__ __device__ strided_iterator operator++(int) {
        strided_iterator tmp(*this);
        operator++();
        return tmp;
    }

    __host__ __device__ strided_iterator& operator--() {
        ptr_ -= stride_;
        return *this;
    }

    __host__ __device__ strided_iterator operator--(int) {
        strided_iterator tmp(*this);
        operator--();
        return tmp;
    }

    __host__ __device__ strided_iterator operator+(size_t n) const {
        return strided_iterator(ptr_ + n * stride_, stride_);
    }

    __host__ __device__ strided_iterator operator-(size_t n) const {
        return strided_iterator(ptr_ - n * stride_, stride_);
    }

    __host__ __device__ strided_iterator& operator+=(size_t n) {
        ptr_ += n * stride_;
        return *this;
    }

    __host__ __device__ strided_iterator& operator-=(size_t n) {
        ptr_ -= n * stride_;
        return *this;
    }

    __host__ __device__ T& operator[](size_t n) const {
        return ptr_[n * stride_];
    }

    __host__ __device__ T& operator*() const { return *ptr_; }

    __host__ __device__ T* operator->() const { return ptr_; }

    __host__ __device__ bool operator==(const strided_iterator& other) const {
        return ptr_ == other.ptr_;
    }

    __host__ __device__ bool operator!=(const strided_iterator& other) const {
        return ptr_ != other.ptr_;
    }

    __host__ __device__ bool operator<(const strided_iterator& other) const {
        return ptr_ < other.ptr_;
    }

    __host__ __device__ bool operator>(const strided_iterator& other) const {
        return ptr_ > other.ptr_;
    }

    __host__ __device__ bool operator<=(const strided_iterator& other) const {
        return ptr_ <= other.ptr_;
    }

    __host__ __device__ bool operator>=(const strided_iterator& other) const {
        return ptr_ >= other.ptr_;
    }

    __host__ __device__ const size_t& stride() const { return stride_; }

    __host__ __device__ size_t operator-(const strided_iterator& other) const {
        return (ptr_ - other.ptr_) / stride_;
    }

    using difference_type   = std::size_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = std::random_access_iterator_tag;

  private:
    T* ptr_;
    size_t stride_;
};

template<class T, uint Rank, class F>
__host__ array<T, Rank - 1>
reduce_last_dim(const array<const T, Rank>& arr, const T& identity, F&& f) {
    const auto& mem_info = arr.memory_info();
    std::vector<int> devices(
        mem_info.devices.get(),
        mem_info.devices.get() + mem_info.num_devices
    );

    std::vector<std::future<void>> futures;
    futures.reserve(devices.size());

    int current_device = cuda::traits::current_device();

    auto device_info = get_device_split_info(arr);

    T result = identity;
    sclx::shape_t<Rank - 1> result_shape;
    for (uint i = 0; i < Rank - 1; ++i) {
        result_shape[i] = arr.shape()[i];
    }

    sclx::array<T, Rank - 1> result_arr(result_shape);
    result_arr.unset_read_mostly();
    sclx::fill(result_arr, identity);
    result_arr.prefetch_async({sclx::cuda::traits::cpu_device_id});

    std::mutex result_mutex;

#ifndef __CLION_IDE__  // CLion shows incorrect errors with thrust library
    for (auto& [_device_id, _slice_idx, _slice_size] : device_info) {
        auto lambda = [&](int device_id, size_t slice_idx, size_t slice_size) {
            cuda::set_device(device_id);

            auto arr_slice
                = arr.get_range({slice_idx}, {slice_idx + slice_size});
            size_t stride = 1;
            for (uint i = 0; i < Rank - 1; ++i) {
                stride *= arr_slice.shape()[i];
            }
            std::vector<T> partial_results(stride);

            sclx::md_index_t<Rank> stride_start_idx;
            sclx::md_index_t<Rank> stride_end_idx;
            stride_end_idx[Rank - 1] = arr_slice.shape()[Rank - 1];
            auto begin_ptr           = &arr_slice[stride_start_idx];
            auto end_ptr             = &arr_slice[stride_end_idx];

            for (size_t s = 0; s < stride; ++s) {
                strided_iterator<const T> begin(begin_ptr + s, stride);
                strided_iterator<const T> end(end_ptr + s, stride);
                partial_results[s]
                    = thrust::reduce(thrust::device, begin, end, identity, f);
            }

            std::lock_guard<std::mutex> lock(result_mutex);
            for (size_t s = 0; s < stride; ++s) {
                result_arr[s] = f(result_arr[s], partial_results[s]);
            }
        };

        futures.emplace_back(std::async(
            std::launch::async,
            lambda,
            _device_id,
            _slice_idx,
            _slice_size
        ));
    }

    cuda::set_device(current_device);

    for (auto& future : futures) {
        future.get();
    }
#endif

    result_arr.set_read_mostly();
    result_arr.prefetch_async(sclx::exec_topology::replicated);

    return result_arr;
}

template<class T, uint Rank, class F>
__host__ array<T, Rank - 1>
reduce_last_dim(const array<T, Rank>& arr, const T& identity, F&& f) {
    return reduce_last_dim(
        static_cast<const array<const T, Rank>&>(arr),
        identity,
        f
    );
}

}  // namespace sclx::algorithm