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

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600 || defined(WIN32)
#warning "This code has been optimized for compute capability 6.0 and above, \
which supports on-demand page migration. If you are using an older compute \
capability, you may experience performance issues. Windows does not \
support on-demand page migration at all."
#endif

#if __CUDA_ARCH__ < 500
static_assert(false, "This code requires compute capability 5.0 or above.");
#endif
#endif

#include "cuda.hpp"
#include "index.cuh"
#include "range.cuh"
#include "shape.cuh"
#include "throw_exception.hpp"
#include <algorithm>
#include <future>
#include <iostream>
#include <memory>
#include <vector>

namespace sclx {

namespace detail {

template<class T>
using dummy_shared_ptr_t = char[sizeof(std::shared_ptr<T>)];

template<class T>
class unified_ptr;

template<class T, class U>
__host__ __device__ unified_ptr<T> static_pointer_cast(const unified_ptr<U>& ptr
);

template<class T, class U>
__host__ __device__ unified_ptr<T>
dynamic_pointer_cast(const unified_ptr<U>& ptr);

template<class T, class U>
__host__ __device__ unified_ptr<T> const_pointer_cast(const unified_ptr<U>& ptr
);

template<class T, class U>
__host__ __device__ unified_ptr<T>
reinterpret_pointer_cast(const unified_ptr<U>& ptr);

template<class T>
class unified_ptr {
  public:
    template<class T_, class U_>
    friend __host__ __device__ unified_ptr<T_>
    static_pointer_cast(const unified_ptr<U_>& ptr);

    template<class T_, class U_>
    friend __host__ __device__ unified_ptr<T_>
    dynamic_pointer_cast(const unified_ptr<U_>& ptr);

    template<class T_, class U_>
    friend __host__ __device__ unified_ptr<T_>
    const_pointer_cast(const unified_ptr<U_>& ptr);

    template<class T_, class U_>
    friend __host__ __device__ unified_ptr<T_>
    reinterpret_pointer_cast(const unified_ptr<U_>& ptr);

    unified_ptr() = default;

    template<class T_ = const T>
    __host__ __device__ operator unified_ptr<T_>() const {
        static_assert(std::is_same_v<const T, T_>, "Invalid cast");
        unified_ptr<T_> ptr;
        ptr.raw_ptr_ = raw_ptr_;
#ifndef __CUDA_ARCH__
        ptr.ptr_ = ptr_;
#endif
        return ptr;
    }

    __host__ __device__ explicit unified_ptr(T* ptr) : raw_ptr_(ptr) {
#ifndef __CUDA_ARCH__
        ptr_ = std::shared_ptr<T>(ptr, [](T* ptr) {
            cudaFree(const_cast<std::remove_const_t<T>*>(ptr));
        });
#endif
    }

    template<class Deleter>
    __host__ __device__ explicit unified_ptr(T* ptr, Deleter deleter)
        : raw_ptr_(ptr) {
#ifndef __CUDA_ARCH__
        ptr_ = std::shared_ptr<T>(ptr, deleter);
#endif
    }

    __host__ __device__ unified_ptr(const unified_ptr<T>& other) {
#ifndef __CUDA_ARCH__
        ptr_ = other.ptr_;
#endif
        raw_ptr_ = other.raw_ptr_;
    }

    __host__ __device__ unified_ptr(unified_ptr<T>&& other) noexcept {
#ifndef __CUDA_ARCH__
        ptr_ = std::move(other.ptr_);
#endif
        raw_ptr_       = other.raw_ptr_;
        other.raw_ptr_ = nullptr;
    }

    __host__ __device__ unified_ptr<T>& operator=(const unified_ptr<T>& other) {
        if (this == &other) {
            return *this;
        }
#ifndef __CUDA_ARCH__
        ptr_ = other.ptr_;
#endif
        raw_ptr_ = other.raw_ptr_;
        return *this;
    }

    __host__ __device__ unified_ptr<T>& operator=(unified_ptr<T>&& other
    ) noexcept {
        if (this == &other) {
            return *this;
        }
#ifndef __CUDA_ARCH__
        ptr_ = std::move(other.ptr_);
#endif
        raw_ptr_       = other.raw_ptr_;
        other.raw_ptr_ = nullptr;
        return *this;
    }

    __host__ __device__ T& operator*() const { return *raw_ptr_; }

    __host__ __device__ T* operator->() const { return raw_ptr_; }

    __host__ __device__ T* get() const { return raw_ptr_; }

    __host__ __device__ explicit operator bool() const {
        return raw_ptr_ != nullptr;
    }

    __host__ auto use_count() const
        -> decltype(std::shared_ptr<T>{}.use_count()) {
#ifndef __CUDA_ARCH__
        return ptr_.use_count();
#else
        return 0;
#endif
    }

    static constexpr auto no_delete = [](T* ptr) {};

    friend class unified_ptr<const T>;
    friend class unified_ptr<std::remove_const_t<T>>;

  private:
    T* raw_ptr_{};
#ifndef __CUDA_ARCH__
    std::shared_ptr<T> ptr_{};
#else
    dummy_shared_ptr_t<T> ptr_{};  // device-safe dummy
#endif
};

template<class T>
__host__ unified_ptr<T> allocate_cuda_usm(size_t size) {
    T* ptr;
    cudaMallocManaged(&ptr, size * sizeof(T));
    return unified_ptr<T>{ptr};
}

template<class T>
__host__ unified_ptr<T> make_unified_ptr(T&& value) {
    T* ptr;
    cudaMallocManaged(&ptr, sizeof(T));
    cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
    return unified_ptr<T>{ptr};
}

template<class T, class U>
__host__ __device__ unified_ptr<T> static_pointer_cast(const unified_ptr<U>& ptr
) {
    unified_ptr<T> new_ptr;
    new_ptr.raw_ptr_ = static_cast<T*>(ptr.raw_ptr_);
#ifndef __CUDA_ARCH__
    new_ptr.ptr_ = std::static_pointer_cast<T>(ptr.ptr_);
#endif

    return new_ptr;
}

template<class T, class U>
__host__ __device__ unified_ptr<T>
dynamic_pointer_cast(const unified_ptr<U>& ptr) {
    unified_ptr<T> new_ptr;
    new_ptr.raw_ptr_ = dynamic_cast<T*>(ptr.raw_ptr_);
#ifndef __CUDA_ARCH__
    new_ptr.ptr_ = std::dynamic_pointer_cast<T>(ptr.ptr_);
#endif

    return new_ptr;
}

template<class T, class U>
__host__ __device__ unified_ptr<T>
reinterpret_pointer_cast(const unified_ptr<U>& ptr) {
    unified_ptr<T> new_ptr;
    new_ptr.raw_ptr_ = reinterpret_cast<T*>(ptr.raw_ptr_);
#ifndef __CUDA_ARCH__
    new_ptr.ptr_ = std::reinterpret_pointer_cast<T>(ptr.ptr_);
#endif

    return new_ptr;
}

template<class T, class U>
__host__ __device__ unified_ptr<T> const_pointer_cast(const unified_ptr<U>& ptr
) {
    unified_ptr<T> new_ptr;
    new_ptr.raw_ptr_ = const_cast<T*>(ptr.raw_ptr_);
#ifndef __CUDA_ARCH__
    new_ptr.ptr_ = std::const_pointer_cast<T>(ptr.ptr_);
#endif

    return new_ptr;
}

}  // namespace detail

enum class data_capture_mode { copy, capture };
enum copy_policy {
    devicedevice = cudaMemcpyDeviceToDevice,
    devicehost   = cudaMemcpyDeviceToHost,
    hostdevice   = cudaMemcpyHostToDevice,
    hosthost     = cudaMemcpyHostToHost
};

template<class T>
struct array_memory_info_t {
    detail::unified_ptr<const T> data_start{};
    detail::unified_ptr<size_t> elements_per_device{};
    detail::unified_ptr<int> devices{};
    uint num_devices{};
};

enum class exec_topology { distributed, replicated };

template<class T, uint Rank>
class array;

template<class T, uint Rank>
__host__ std::vector<std::tuple<int, size_t, size_t>>
get_device_split_info(const array<T, Rank>& arr);

template<class T, uint Rank>
class array {
  public:
    using mem_info_t = array_memory_info_t<std::remove_const_t<T>>;

    array() = default;

    __host__ array(std::initializer_list<size_t> shape) : shape_(shape) {
        data_ = detail::allocate_cuda_usm<T>(elements());
        set_primary_devices();
    }

    __host__ explicit array(const shape_t<Rank>& shape) : shape_(shape) {
        data_ = detail::allocate_cuda_usm<T>(elements());
        set_primary_devices();
    }

#ifndef __CUDA_ARCH__
    __host__ array(
        const shape_t<Rank>& shape,
        const detail::unified_ptr<const T>& data,
        data_capture_mode mode = data_capture_mode::copy,
        copy_policy policy     = copy_policy::hostdevice
    )
        : shape_(shape) {
        if (mode == data_capture_mode::copy) {
            data_ = detail::allocate_cuda_usm<T>(elements());
            cudaMemcpy(
                const_cast<std::remove_const_t<T>*>(data_.get()),
                data.get(),
                elements() * sizeof(T),
                static_cast<cudaMemcpyKind>(policy)
            );
        } else {
            auto ptr = std::make_shared<const float>(1.f);
            data_    = detail::const_pointer_cast<T>(std::move(data));
        }
    }
#else

    __device__
    array(const shape_t<Rank>& shape, const detail::unified_ptr<const T>& data)
        : shape_(shape),
          data_(detail::unified_ptr<T>{const_cast<T*>(data.get())}) {}
#endif

    template<class T_ = const T>
    __host__ __device__
    operator array<T_, Rank>() const {  // NOLINT(google-explicit-constructor)
        static_assert(std::is_same_v<const T, T_>, "Invalid cast");
        array<const T, Rank> new_arr;
        new_arr.shape_       = shape_;
        new_arr.data_        = data_;
        new_arr.memory_info_ = memory_info_;
        return new_arr;
    }

    __host__ __device__ T& operator[](const md_index_t<Rank>& index) const {
        return data_.get()[index.as_linear(shape_)];
    }

    __host__ __device__ T& operator[](index_t index) const {
        return data_.get()[index];
    }

    template<class... Args>
    __host__ __device__ T& operator()(const Args&... args) const {
        return this->operator[](md_index_t<Rank>{static_cast<size_t>(args)...});
    }

    __host__ __device__ const shape_t<Rank>& shape() const { return shape_; }

    __host__ __device__ detail::unified_ptr<const T> data() const {
        return data_;
    }

    __host__ __device__ T* begin() const { return data_.get(); }

    __host__ __device__ T* end() const { return data_.get() + elements(); }

    __host__ __device__ size_t elements() const { return shape_.elements(); }

    [[nodiscard]] __host__ __device__ constexpr uint rank() const {
        return Rank;
    }

    __host__ array& set_primary_devices() {
        int device_count = cuda::traits::device_count();
        std::vector<int> devices(device_count);
        for (int i = 0; i < device_count; ++i) {
            devices[i] = i;
        }
        const auto& old_buf = std::cerr.rdbuf();
        std::cerr.rdbuf(nullptr);
        set_primary_devices(devices);
        std::cerr.rdbuf(old_buf);

        return *this;
    }

    __host__ array& set_primary_devices(const std::vector<int>& devices) {
        size_t sig_dim       = shape_[Rank - 1];
        size_t sig_dim_split = (sig_dim + devices.size() - 1) / devices.size();
        shape_t<Rank> split_shape;
        if constexpr (Rank > 1) {
            for (uint i = 0; i < Rank - 1; ++i) {
                split_shape[i] = shape_[i];
            }
        }
        split_shape[Rank - 1]      = sig_dim_split;
        size_t elements_per_device = split_shape.elements();
        uint max_devices
            = (elements() + elements_per_device - 1) / elements_per_device;
        std::vector<int> devices_to_use = devices;
        if (devices.size() > max_devices) {
            std::cerr << "Warning: too many devices specified for array. "
                      << "Using " << max_devices << " devices instead."
                      << std::endl;
            devices_to_use.resize(max_devices);
        }

        auto max_dev_id  = *std::max_element(devices.begin(), devices.end());
        int device_count = cuda::traits::device_count();
        if (max_dev_id >= device_count) {
            throw_exception<std::invalid_argument>(
                "Invalid device list; device ids must be less than the number "
                "of devices, which is "
                    + std::to_string(device_count) + ".",
                "sclx::array::"
            );
        }

        std::vector<std::tuple<int, size_t, size_t>> device_split_info;
        for (uint d = 0; d < devices_to_use.size(); ++d) {
            size_t slice_idx = d * sig_dim_split;
            size_t slice_len = std::min(sig_dim - slice_idx, sig_dim_split);
            device_split_info
                .emplace_back(devices_to_use[d], slice_idx, slice_len);
        }

        set_primary_devices(device_split_info);

        return *this;
    };

    // split info is a vector of tuples of the form (device_id, slice_idx,
    // slice_len) where we slice the array along the last dimension
    __host__ array& set_primary_devices(
        const std::vector<std::tuple<int, size_t, size_t>>& device_split_info
    ) {
        if (memory_info_.get() == nullptr) {
            memory_info_ = detail::make_unified_ptr<mem_info_t>({});
        }
        if (is_slice()) {
            throw_exception<std::runtime_error>(
                "Array is a get_slice and cannot have its primary devices "
                "changed.",
                "sclx::array::"
            );
        }
        memory_info_->data_start = data_;
        memory_info_->elements_per_device
            = detail::allocate_cuda_usm<size_t>(device_split_info.size());
        memory_info_->devices
            = detail::allocate_cuda_usm<int>(device_split_info.size());
        memory_info_->num_devices = device_split_info.size();

        for (int d = 0; d < device_split_info.size(); ++d) {
            const auto& [device_id, slice_idx, slice_range]
                = device_split_info[d];
            sclx::md_index_t<Rank> start_idx;
            start_idx[Rank - 1] = slice_idx;
            sclx::md_index_t<Rank> end_idx;
            end_idx[Rank - 1]        = slice_idx + slice_range;
            auto elements_per_device = std::distance(
                &(this->operator[](start_idx)),
                &(this->operator[](end_idx))
            );
            memory_info_->elements_per_device.get()[d] = elements_per_device;
            memory_info_->devices.get()[d]             = device_id;
        }

        int device_count = cuda::traits::device_count();
        for (int d = 0; d < device_count; ++d) {
            cuda::mem_advise(
                data_.get(),
                elements() * sizeof(T),
                cuda::traits::mem_advise::unset_accessed_by,
                d
            );
            cuda::mem_advise(
                data_.get(),
                elements() * sizeof(T),
                cuda::traits::mem_advise::unset_preferred_location,
                d
            );
        }

        size_t offset = 0;
        for (int d = 0; d < device_split_info.size(); ++d) {
            const auto& [device_id, slice_idx, slice_range]
                = device_split_info[d];
            cuda::mem_advise(
                data_.get(),
                elements() * sizeof(T),
                cuda::traits::mem_advise::set_accessed_by,
                device_id
            );
            cuda::mem_advise(
                data_.get() + offset,
                memory_info_->elements_per_device.get()[d] * sizeof(T),
                cuda::traits::mem_advise::set_preferred_location,
                device_id
            );
            offset += memory_info_->elements_per_device.get()[d];
        }

        cuda::mem_advise(
            data_.get(),
            elements() * sizeof(T),
            cuda::traits::mem_advise::set_accessed_by,
            cuda::traits::cpu_device_id
        );

        set_read_mostly();

        prefetch_async(exec_topology::distributed);

        return *this;
    }

    __host__ array& set_read_mostly() {
        cuda::mem_advise(
            data_.get(),
            elements() * sizeof(T),
            cuda::traits::mem_advise::set_read_mostly,
            0
        );
        return *this;
    }

    __host__ array& unset_read_mostly() {
        cuda::mem_advise(
            data_.get(),
            elements() * sizeof(T),
            cuda::traits::mem_advise::unset_read_mostly,
            0
        );
        return *this;
    }

    __host__ std::future<void> prefetch_async(
        const std::vector<int>& device_ids,
        exec_topology topology = exec_topology::replicated
    ) {
        if (topology == exec_topology::distributed) {
            return prefetch_async_distributed(device_ids);
        } else if (topology == exec_topology::replicated) {
            return prefetch_async_replicated(device_ids);
        } else {
            throw_exception<std::invalid_argument>(
                "Invalid exec_topology specified.",
                "sclx::array::"
            );

            return std::future<void>{};  // suppress compiler warning
        }
    }

    __host__ std::future<void>
    prefetch_async(exec_topology topology = exec_topology::replicated) {
        std::vector<cuda::stream_t> streams;
        if (topology == exec_topology::distributed) {
            auto device_info = get_device_split_info(*this);
            for (auto& [device_id, slice_idx, slice_size] : device_info) {
                md_index_t<Rank> start_idx{};
                start_idx[Rank - 1] = slice_idx;
                md_index_t<Rank> end_idx{};
                end_idx[Rank - 1] = slice_idx + slice_size;

                int stream_id = (device_id != cuda::traits::cpu_device_id)
                                  ? device_id
                                  : 0;
                auto stream   = cuda::stream_t::create_for_device(stream_id);
                cuda::mem_prefetch_async(
                    data_.get() + start_idx.as_linear(shape_),
                    data_.get() + end_idx.as_linear(shape_),
                    device_id,
                    stream
                );

                streams.emplace_back(std::move(stream));
            }

            return std::async(
                std::launch::async,
                [streams = std::move(streams)]() {
                    for (const auto& stream : streams) {
                        stream.synchronize();
                    }
                }
            );
        } else if (topology == exec_topology::replicated) {
            return prefetch_async_replicated(std::vector<int>(
                memory_info_->devices.get(),
                memory_info_->devices.get() + memory_info_->num_devices
            ));
        } else {
            throw_exception<std::invalid_argument>(
                "Invalid execution topology.",
                "sclx::array::"
            );

            return {};  // Silence compiler warning
        }
    }

    const mem_info_t& memory_info() const { return *memory_info_; }

    __host__ __device__ bool is_slice() const {
        return memory_info_->data_start.get() != data_.get()
            && memory_info_->data_start.get() != nullptr;
    }

    template<uint IndexRank>
    __host__ __device__ array<T, Rank - IndexRank>
    get_slice(md_index_t<IndexRank> slice_idx) const {
        md_index_t<IndexRank - 1> next_slice;
        for (uint i = 0; i < IndexRank - 1; ++i) {
            next_slice[i] = slice_idx[i + 1];
        }
        array<T, Rank - IndexRank + 1> partial_slice = get_slice(next_slice);

        md_index_t<Rank - IndexRank + 1> start_index;
        start_index[Rank - IndexRank] = slice_idx[0];
        detail::unified_ptr<T> data(
            &partial_slice[start_index],
            detail::unified_ptr<T>::no_delete
        );

        shape_t<Rank - IndexRank> shape;
        for (uint i = 0; i < Rank - IndexRank; ++i) {
            shape[i] = partial_slice.shape()[i];
        }

#ifdef __CUDA_ARCH__
        return array<T, Rank - IndexRank>{shape, data};
#else
        array<T, Rank - IndexRank> arr{shape, data, data_capture_mode::capture};
        arr.memory_info_ = partial_slice.memory_info_;
        return arr;
#endif
    }

    __host__ __device__ auto get_slice(md_index_t<1> slice_idx) const {
        if constexpr (Rank == 1) {
            return this->operator[](slice_idx);
        } else {

            md_index_t<Rank> start_index;
            start_index[Rank - 1] = slice_idx[0];

            detail::unified_ptr<T> data(
                &(*this)[start_index],
                detail::unified_ptr<T>::no_delete
            );

            shape_t<Rank - 1> shape;
            if constexpr (Rank > 1) {
                for (uint i = 0; i < Rank - 1; ++i) {
                    shape[i] = this->shape()[i];
                }
            }

#ifdef __CUDA_ARCH__
            return array<T, Rank - 1>{shape, data};
#else
            array<T, Rank - 1> arr{shape, data, data_capture_mode::capture};
            arr.memory_info_ = memory_info_;
            return arr;
#endif
        }
    }

    __host__ __device__ array
    get_range(md_index_t<1> start, md_index_t<1> end) const {
        md_index_t<Rank> start_index;
        start_index[Rank - 1] = start[0];

        detail::unified_ptr<T> data(
            &(*this)[start_index],
            detail::unified_ptr<T>::no_delete
        );

        shape_t<Rank> shape;
        if constexpr (Rank > 1) {
            for (uint i = 0; i < Rank - 1; ++i) {
                shape[i] = this->shape()[i];
            }
        }
        shape[Rank - 1] = end[0] - start[0];

#ifdef __CUDA_ARCH__
        return array{shape, data};
#else
        array arr{shape, data, data_capture_mode::capture};
        arr.memory_info_ = memory_info_;
        return arr;
#endif
    }

    template<class T_, uint Rank_>
    friend class array;

  private:
    detail::unified_ptr<T> data_{};
    shape_t<Rank> shape_{};

    detail::unified_ptr<mem_info_t> memory_info_{};

    std::future<void>
    prefetch_async_distributed(const std::vector<int>& device_ids) {
        std::vector<cuda::stream_t> streams;
        shape_t<Rank> split_shape{};
        split_shape[Rank - 1]
            = (shape_[Rank - 1] + device_ids.size() - 1) / device_ids.size();
        size_t split_size = split_shape.elements();
        for (auto& device_id : device_ids) {
            int stream_id
                = (device_id != cuda::traits::cpu_device_id) ? device_id : 0;
            auto stream = cuda::stream_t::create_for_device(stream_id);

            cuda::mem_prefetch_async(
                data_.get() + device_id * split_size,
                data_.get() + (device_id + 1) * split_size,
                device_id,
                stream
            );
            streams.emplace_back(std::move(stream));
        }

        return std::async(std::launch::async, [streams = std::move(streams)]() {
            for (const auto& stream : streams) {
                stream.synchronize();
            }
        });
    }

    std::future<void>
    prefetch_async_replicated(const std::vector<int>& device_ids) {
        std::vector<cuda::stream_t> streams;
        for (auto& device_id : device_ids) {
            int stream_id
                = (device_id != cuda::traits::cpu_device_id) ? device_id : 0;
            auto stream = cuda::stream_t::create_for_device(stream_id);

            cuda::mem_prefetch_async(
                this->begin(),
                this->end(),
                device_id,
                stream
            );
            streams.emplace_back(std::move(stream));
        }

        return std::async(std::launch::async, [streams = std::move(streams)]() {
            for (const auto& stream : streams) {
                stream.synchronize();
            }
        });
    }
};

template<class T, uint RankOld, uint RankNew>
array<T, RankNew> __host__ __device__
reshape(const array<T, RankOld>& arr, const shape_t<RankNew>& shape) {
    if (arr.shape().elements() != shape.elements()) {
        throw_exception<std::invalid_argument>(
            "Cannot reshape array with different number of elements"
        );
    }
#ifdef __CUDA_ARCH__
    return array<T, RankNew>{shape, arr.data()};
#else
    return array<T, RankNew>{shape, arr.data(), data_capture_mode::copy};
#endif
}

template<class T, uint Rank>
__host__ __device__ array<T, 1> flatten(const array<T, Rank>& arr) {
#ifdef __CUDA_ARCH__
    return array<T, 1>{{arr.shape().elements()}, arr.data()};
#else
    return array<T, 1>{
        {arr.shape().elements()},
        arr.data(),
        data_capture_mode::copy};
#endif
}

/**
 * @brief Get device split info for a given array
 *
 * Note that all arrays are split among their devices along the most significant
 * dimension. This ensures that data in lower dimensions is contiguous and
 * on the same device. Therefore, the split info returned by this function is
 * a vector of tuples, one for each device the array is split across. The first
 * element of the tuple is the device id, the second is the split index along
 * the most significant dimension, and the third is the range along
 * the most significant dimension that is on that device.
 *
 * For example, if the array is split across devices 0 and 1, and had a shape of
 * (3, 100) then the returned vector would be:
 *
 *    ``[(0, 0, 50), (1, 50, 100)]``
 *
 * @tparam T type of array
 * @tparam Rank rank of array
 * @param arr array to get split info for
 * @return vector of tuples containing split info
 */
template<class T, uint Rank>
__host__ std::vector<std::tuple<int, size_t, size_t>>
get_device_split_info(const array<T, Rank>& arr) {
    std::vector<std::tuple<int, size_t, size_t>> splits;

    auto& mem_info           = arr.memory_info();
    int start_device         = 0;
    const T* device_data_ptr = mem_info.data_start.get();
    for (int d = 0; d < mem_info.num_devices; ++d) {
        if (device_data_ptr < arr.data().get()) {
            device_data_ptr += mem_info.elements_per_device.get()[d];
        } else {
            start_device = d;
            break;
        }
    }
    size_t num_elements_to_next_device
        = mem_info.elements_per_device.get()[start_device]
        - std::distance(device_data_ptr, arr.data().get());

    sclx::md_index_t<Rank> start_idx;
    auto end_idx = md_index_t<Rank>::create_from_linear(
        std::min(
            num_elements_to_next_device + start_idx.as_linear(arr.shape()),
            arr.elements()
        ),
        arr.shape()
    );
    splits.push_back(
        {mem_info.devices.get()[start_device],
         start_idx[Rank - 1],
         end_idx[Rank - 1] - start_idx[Rank - 1]}
    );
    num_elements_to_next_device
        = mem_info.elements_per_device.get()[start_device + 1];
    while (end_idx.as_linear(arr.shape()) < arr.elements()) {
        start_idx    = end_idx;
        start_device = start_device + 1;
        end_idx      = sclx::md_index_t<Rank>::create_from_linear(
            std::min(
                num_elements_to_next_device + start_idx.as_linear(arr.shape()),
                arr.elements()
            ),
            arr.shape()
        );
        splits.push_back(
            {mem_info.devices.get()[start_device],
             start_idx[Rank - 1],
             end_idx[Rank - 1] - start_idx[Rank - 1]}
        );
    }
    return splits;
}

template class array<float, 1>;

}  // namespace sclx