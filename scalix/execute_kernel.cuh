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
#include "array.cuh"
#include "cuda.hpp"
#include <future>
#include <mutex>

namespace sclx {

namespace detail {

struct _debug_kernel_mutex {
    static std::mutex mutex;
};
#ifdef SCALIX_DEBUG_KERNEL_LAUNCH
constexpr bool _debug_kernel_launch = true;
std::mutex _debug_kernel_mutex::mutex{};
#else
constexpr bool _debug_kernel_launch = false;
#endif

template<class F, uint RangeRank>
__global__ void sclx_kernel(
    F f,
    sclx::md_range_t<RangeRank> local_range,
    sclx::md_range_t<RangeRank> global_range,
    sclx::md_index_t<RangeRank> start_idx
) {
    auto idx = sclx::md_index_t<RangeRank>::from_flat_index(
        blockIdx.x * blockDim.x + threadIdx.x
            + start_idx.flat_index(global_range),
        global_range
    );
    while (idx.flat_index(global_range)
           < start_idx.flat_index(global_range) + local_range.elements()) {
        f(idx);
        idx = sclx::md_index_t<RangeRank>::from_flat_index(
            idx.flat_index(global_range) + gridDim.x * blockDim.x,
            global_range
        );
    }
}
struct default_kernel_tag {
    template<uint RangeRank, uint ThreadBlockRank, class F>
    __host__ static void execute(
        const std::vector<std::tuple<int, size_t, size_t>>& device_info,
        const sclx::md_range_t<RangeRank>& global_range,
        const sclx::shape_t<ThreadBlockRank>& block_shape,
        size_t grid_size,
        size_t local_mem_size,
        F&& f
    ) {
        std::vector<sclx::cuda::stream_t> streams;
        for (auto& [device_id, slice_idx, slice_size] : device_info) {
            streams.emplace_back(
                sclx::cuda::stream_t::create_for_device(device_id)
            );
        }

        if constexpr (sclx::detail::_debug_kernel_launch) {
            sclx::detail::_debug_kernel_mutex::mutex.lock();
            std::cout << "Launching Scalix Kernel" << std::endl;
        }

        int iter = 0;
        for (auto& [device_id, slice_idx, slice_size] : device_info) {
            sclx::cuda::set_device(device_id);  // init context

            sclx::md_range_t<RangeRank> device_range{};
            if constexpr (RangeRank > 1) {
                for (uint i = 0; i < RangeRank - 1; i++) {
                    device_range[i] = global_range[i];
                }
            }
            device_range[RangeRank - 1] = slice_size;

            sclx::md_index_t<RangeRank> device_start_idx{};
            device_start_idx[RangeRank - 1] = slice_idx;

            size_t total_threads = device_range.elements();
            size_t max_grid_size = (total_threads + block_shape.elements() - 1)
                                 / block_shape.elements();
            grid_size = std::min(max_grid_size, grid_size);

            auto& stream = streams[iter++];

            if constexpr (sclx::detail::_debug_kernel_launch) {
                std::cout << "  Launching kernel on device " << device_id
                          << std::endl;
                std::cout << "      Device range: " << device_range
                          << std::endl;
                std::cout << "      Device start index: " << device_start_idx
                          << std::endl;
                std::cout << "      Global range: " << global_range
                          << std::endl;
            }
            sclx_kernel<<<
                grid_size,
                block_shape.elements(),
                local_mem_size,
                stream>>>(f, device_range, global_range, device_start_idx);
        }

        if constexpr (sclx::detail::_debug_kernel_launch) {
            std::cout << "Kernel launch complete" << std::endl << std::endl;
            sclx::detail::_debug_kernel_mutex::mutex.unlock();
        }

        for (auto& stream : streams) {
            stream.synchronize();
        }
    }
};
}  // namespace detail

template<class T, uint Rank, uint N>
class array_list {
  public:
    __host__ array_list(std::initializer_list<array<T, Rank>> arrays) {
        if (arrays.size() != N) {
            throw_exception<std::invalid_argument>(
                "array_list must be initialized with exactly N arrays",
                "sclx::array_list::"
            );
        }
        std::copy(arrays.begin(), arrays.end(), arrays_);
        shape_ = arrays_[0].shape();
        for (auto& array : arrays_) {
            if (array.shape() != shape_) {
                throw_exception<std::invalid_argument>(
                    "All arrays in array_list must have the same shape",
                    "sclx::array_list::"
                );
            }
        }
    }

    __host__ explicit array_list(const std::vector<array<T, Rank>>& arrays) {
        if (arrays.size() != N) {
            throw_exception<std::invalid_argument>(
                "array_list must be initialized with exactly N arrays",
                "sclx::array_list::"
            );
        }
        std::copy(arrays.begin(), arrays.end(), arrays_);
        shape_ = arrays_[0].shape();
        for (auto& array : arrays_) {
            if (array.shape() != shape_) {
                throw_exception<std::invalid_argument>(
                    "All arrays in array_list must have the same shape",
                    "sclx::array_list::"
                );
            }
        }
    }

    __host__ explicit array_list(const std::array<array<T, Rank>, N>& arrays) {
        std::copy(arrays.begin(), arrays.end(), arrays_);
        shape_ = arrays_[0].shape();
        for (auto& array : arrays_) {
            if (array.shape() != shape_) {
                throw_exception<std::invalid_argument>(
                    "All arrays in array_list must have the same shape",
                    "sclx::array_list::"
                );
            }
        }
    }

    __host__ __device__ array<T, Rank>& operator[](size_t i) {
        return arrays_[i];
    }

    __host__ __device__ const array<T, Rank>& operator[](size_t i) const {
        return arrays_[i];
    }

    __host__ __device__ shape_t<Rank> shape() const { return shape_; }

    __host__ __device__ array<T, Rank>* begin() { return arrays_; }

    __host__ __device__ array<T, Rank>* end() { return arrays_ + N; }

    __host__ __device__ const array<T, Rank>* begin() const { return arrays_; }

    __host__ __device__ const array<T, Rank>* end() const {
        return arrays_ + N;
    }

  private:
    array<T, Rank> arrays_[N];
    shape_t<Rank> shape_;
};

class kernel_handler {
  public:
    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        uint ThreadBlockRank,
        class T,
        uint ResultRank,
        class F>
    __host__ void launch(
        md_range_t<RangeRank> range,
        array<T, ResultRank> result,
        shape_t<ThreadBlockRank> block_shape,
        size_t grid_size,
        F&& f
    ) const {
        if (range[RangeRank - 1] != result.shape()[ResultRank - 1]) {
            throw std::runtime_error(
                "Range and result last dimension must be the same"
            );
        }

        auto device_info = get_device_split_info(result);

        result.unset_read_mostly();
        KernelTag::template execute<RangeRank, ThreadBlockRank>(
            device_info,
            range,
            block_shape,
            grid_size,
            local_mem_size_,
            f
        );
        result.set_read_mostly();
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        class T,
        uint ResultRank,
        class F>
    __host__ void
    launch(md_range_t<RangeRank> range, array<T, ResultRank> result, F&& f)
        const {
        launch<KernelTag>(
            range,
            result,
            cuda::traits::kernel::default_block_shape,
            cuda::traits::kernel::default_grid_size,
            f
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        uint ThreadBlockRank,
        class T,
        uint ResultRank,
        uint NResult,
        class F>
    __host__ void launch(
        md_range_t<RangeRank> range,
        array_list<T, ResultRank, NResult> result,
        shape_t<ThreadBlockRank> block_shape,
        size_t grid_size,
        F&& f
    ) const {
        if (range[RangeRank - 1] != result.shape()[ResultRank - 1]) {
            throw std::runtime_error(
                "Range and result last dimension must be the same"
            );
        }

        for (auto& array : result) {
            array.unset_read_mostly();
        }

        auto device_info = get_device_split_info(result[0]);

        KernelTag::template execute<RangeRank, ThreadBlockRank>(
            device_info,
            range,
            block_shape,
            grid_size,
            local_mem_size_,
            f
        );

        for (auto& array : result) {
            array.set_read_mostly();
        }
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        class T,
        uint ResultRank,
        uint NResult,
        class F>
    __host__ void launch(
        md_range_t<RangeRank> range,
        array_list<T, ResultRank, NResult> result,
        F&& f
    ) const {
        launch<KernelTag>(
            range,
            result,
            cuda::traits::kernel::default_block_shape,
            cuda::traits::kernel::default_grid_size,
            f
        );
    }

    template<uint ThreadBlockRank>
    __device__ md_index_t<ThreadBlockRank>
    get_local_thread_idx(const shape_t<ThreadBlockRank>& block_shape) const {
        return md_index_t<ThreadBlockRank>::from_flat_index(
            threadIdx.x,
            block_shape
        );
    }

    __device__ md_index_t<1> get_local_thread_idx() const {
        return {threadIdx.x};
    }

    __device__ md_index_t<1> get_device_block_idx() const {
        return {blockIdx.x};
    }

    __device__ void sync_threads() const { __syncthreads(); }

    template<class T, uint Rank>
    friend class local_array;

  private:
    __device__ static char* get_local_mem(const size_t& offset) {
        extern __shared__ char local_mem[];
        return local_mem + offset;
    }

    size_t local_mem_size_ = 0;
};

template<class T, uint Rank>
class local_array {
  public:
    __host__ local_array(kernel_handler& handler, const shape_t<Rank>& shape)
        : shape_(shape) {
        offset_ = handler.local_mem_size_;
        handler.local_mem_size_ += elements() * sizeof(T);
    }

    __device__ T& operator[](const md_index_t<Rank>& idx) {
        if (!data_) {
            data_
                = reinterpret_cast<T*>(kernel_handler::get_local_mem(offset_));
        }
        return data_[idx.flat_index(shape_)];
    }

    __host__ __device__ size_t elements() const { return shape_.elements(); }

    __host__ __device__ const shape_t<Rank>& shape() const { return shape_; }

  private:
    size_t offset_;
    shape_t<Rank> shape_;
    T* data_{};  // will be set by the sclx_kernel
};

template<class F>
__host__ std::future<void> execute_kernel(F&& f) {
    return std::async(std::launch::async, [=]() {
        cuda::set_device(0);  // initialize cuda
        kernel_handler handler;
        f(handler);
    });
}

}  // namespace sclx

#define REGISTER_SCALIX_KERNEL_TAG(Tag)                                        \
    template<class F, uint RangeRank>                                          \
    __global__ void sclx_##Tag(                                                \
        F f,                                                                   \
        sclx::md_range_t<RangeRank> local_range,                               \
        sclx::md_range_t<RangeRank> global_range,                              \
        sclx::md_index_t<RangeRank> start_idx                                  \
    ) {                                                                        \
        auto idx = sclx::md_index_t<RangeRank>::from_flat_index(               \
            blockIdx.x * blockDim.x + threadIdx.x                              \
                + start_idx.flat_index(global_range),                          \
            global_range                                                       \
        );                                                                     \
        while (idx.flat_index(global_range)                                    \
               < start_idx.flat_index(global_range) + local_range.elements()   \
        ) {                                                                    \
            f(idx);                                                            \
            idx = sclx::md_index_t<RangeRank>::from_flat_index(                \
                idx.flat_index(global_range) + gridDim.x * blockDim.x,         \
                global_range                                                   \
            );                                                                 \
        }                                                                      \
    }                                                                          \
    struct Tag {                                                               \
        template<uint RangeRank, uint ThreadBlockRank, class F>                \
        __host__ static void execute(                                          \
            const std::vector<std::tuple<int, size_t, size_t>>& device_info,   \
            const sclx::md_range_t<RangeRank>& global_range,                   \
            const sclx::shape_t<ThreadBlockRank>& block_shape,                 \
            size_t grid_size,                                                  \
            size_t local_mem_size,                                             \
            F&& f                                                              \
        ) {                                                                    \
            std::vector<sclx::cuda::stream_t> streams;                         \
            for (auto& [device_id, slice_idx, slice_size] : device_info) {     \
                streams.emplace_back(                                          \
                    sclx::cuda::stream_t::create_for_device(device_id)         \
                );                                                             \
            }                                                                  \
                                                                               \
            if constexpr (sclx::detail::_debug_kernel_launch) {                \
                sclx::detail::_debug_kernel_mutex::mutex.lock();               \
                std::cout << "Launching Scalix Kernel" << std::endl;           \
            }                                                                  \
                                                                               \
            int iter = 0;                                                      \
            for (auto& [device_id, slice_idx, slice_size] : device_info) {     \
                sclx::cuda::set_device(device_id);                             \
                sclx::md_range_t<RangeRank> device_range{};                    \
                if constexpr (RangeRank > 1) {                                 \
                    for (uint i = 0; i < RangeRank - 1; i++) {                 \
                        device_range[i] = global_range[i];                     \
                    }                                                          \
                }                                                              \
                device_range[RangeRank - 1] = slice_size;                      \
                                                                               \
                sclx::md_index_t<RangeRank> device_start_idx{};                \
                device_start_idx[RangeRank - 1] = slice_idx;                   \
                                                                               \
                size_t total_threads = device_range.elements();                \
                size_t max_grid_size                                           \
                    = (total_threads + block_shape.elements() - 1)             \
                    / block_shape.elements();                                  \
                grid_size = std::min(max_grid_size, grid_size);                \
                                                                               \
                auto& stream = streams[iter++];                                \
                                                                               \
                if constexpr (sclx::detail::_debug_kernel_launch) {            \
                    std::cout << "  Launching kernel on device " << device_id  \
                              << std::endl;                                    \
                    std::cout << "      Device range: " << device_range        \
                              << std::endl;                                    \
                    std::cout                                                  \
                        << "      Device start index: " << device_start_idx    \
                        << std::endl;                                          \
                    std::cout << "      Global range: " << global_range        \
                              << std::endl;                                    \
                }                                                              \
                sclx_##Tag<<<                                                  \
                    grid_size,                                                 \
                    block_shape.elements(),                                    \
                    local_mem_size,                                            \
                    stream>>>(                                                 \
                    f,                                                         \
                    device_range,                                              \
                    global_range,                                              \
                    device_start_idx                                           \
                );                                                             \
                                                                               \
                streams.push_back(stream);                                     \
            }                                                                  \
                                                                               \
            if constexpr (sclx::detail::_debug_kernel_launch) {                \
                std::cout << "Kernel launch complete" << std::endl             \
                          << std::endl;                                        \
                sclx::detail::_debug_kernel_mutex::mutex.unlock();             \
            }                                                                  \
                                                                               \
            for (auto& stream : streams) {                                     \
                stream.synchronize();                                          \
            }                                                                  \
        }                                                                      \
    }
