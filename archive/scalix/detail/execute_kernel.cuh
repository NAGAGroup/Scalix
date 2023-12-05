
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
#include "../cuda.hpp"
#include "../kernel_info.cuh"
#include <iostream>
#include <mutex>
#include <vector>

namespace sclx::detail {

struct _debug_kernel_mutex {
    static std::mutex mutex;
};
#ifdef SCALIX_DEBUG_KERNEL_LAUNCH
constexpr bool _debug_kernel_launch = true;
std::mutex _debug_kernel_mutex::mutex{};
#else
constexpr bool _debug_kernel_launch = false;
#endif

template<class F, uint RangeRank, uint ThreadBlockRank>
__global__ void
sclx_kernel(F f, sclx::kernel_info<RangeRank, ThreadBlockRank> metadata) {
    const sclx::md_range_t<RangeRank>& global_range = metadata.global_range();
    sclx::md_index_t<RangeRank> idx = metadata.global_thread_id();
    while (idx.as_linear(global_range)
           < metadata.start_index().as_linear(global_range)
                 + metadata.device_range().elements()) {
        f(idx, metadata);

        idx = sclx::md_index_t<RangeRank>::create_from_linear(
            idx.as_linear(global_range) + metadata.grid_stride(),
            global_range
        );
        metadata.increment_stride_count();
    }
}

template<class F, class IndexGenerator, uint ThreadBlockRank>
__global__ void sclx_kernel(
    F f,
    sclx::kernel_info<IndexGenerator::range_rank, ThreadBlockRank> metadata,
    IndexGenerator idx_gen,
    sclx::index_t slice_idx,
    sclx::index_t slice_size
) {
    constexpr uint range_rank = IndexGenerator::range_rank;
    constexpr uint index_rank = IndexGenerator::index_rank;
    const sclx::md_range_t<range_rank>& global_range = metadata.global_range();
    sclx::md_index_t<range_rank> global_thread_id = metadata.global_thread_id();

    while (global_thread_id.as_linear(global_range) < global_range.elements()) {
        sclx::md_index_t<index_rank> idx = idx_gen(global_thread_id);

        if (idx[index_rank - 1] >= slice_idx
            && idx[index_rank - 1] < slice_idx + slice_size) {
            f(idx, metadata);
        }

        global_thread_id = sclx::md_index_t<>::create_from_linear(
            global_thread_id.as_linear(global_range) + metadata.grid_stride(),
            global_range
        );
        metadata.increment_stride_count();
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
            sclx::cuda::set_device(device_id); /* init context */

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

            sclx::kernel_info<RangeRank, ThreadBlockRank> metadata(
                global_range,
                device_range,
                block_shape,
                device_start_idx,
                iter
            );

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

            auto kernel
                = &sclx_kernel<std::decay_t<F>, RangeRank, ThreadBlockRank>;

            uint forty_eight_kb = 1024 * 48;
            uint maxbytes
                = std::max(forty_eight_kb, static_cast<uint>(local_mem_size));
            auto raw_error = cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxbytes
            );
            sclx::cuda::cuda_exception::raise_if_not_success(
                raw_error,
                std::experimental::source_location(),
                "sclx::detail::"
            );

            kernel<<<
                grid_size,
                block_shape.elements(),
                local_mem_size,
                stream>>>(f, metadata);
        }

        if constexpr (sclx::detail::_debug_kernel_launch) {
            std::cout << "Kernel launch complete" << std::endl << std::endl;
            sclx::detail::_debug_kernel_mutex::mutex.unlock();
        }

        for (auto& stream : streams) {
            stream.synchronize();
        }
    }

    template<class IndexGenerator, uint ThreadBlockRank, class F>
    __host__ static void execute(
        const std::vector<std::tuple<int, size_t, size_t>>& device_info,
        IndexGenerator&& index_generator,
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

        size_t total_slice_size = 0;
        if constexpr (sclx::detail::_debug_kernel_launch) {
            sclx::detail::_debug_kernel_mutex::mutex.lock();
            std::cout << "Launching Scalix Kernel using IndexGenerator"
                      << std::endl;
            for (auto& [device_id, slice_idx, slice_size] : device_info) {
                total_slice_size += slice_size;
            }
        }

        int iter = 0;
        for (auto& [device_id, slice_idx, slice_size] : device_info) {
            sclx::cuda::set_device(device_id); /* init context */

            size_t total_threads = index_generator.range().elements();
            size_t max_grid_size = (total_threads + block_shape.elements() - 1)
                                 / block_shape.elements();
            grid_size = std::min(max_grid_size, grid_size);

            constexpr uint range_rank
                = std::remove_reference_t<IndexGenerator>::range_rank;
            sclx::kernel_info<range_rank, ThreadBlockRank> metadata(
                index_generator.range(),
                index_generator.range(),
                block_shape,
                {},
                iter
            );

            auto& stream = streams[iter++];

            if constexpr (sclx::detail::_debug_kernel_launch) {
                std::cout << "  Launching kernel on device " << device_id
                          << std::endl;
                std::cout << "      Device slice size: " << slice_size
                          << std::endl;
                std::cout << "      Device slice start index: " << slice_idx
                          << std::endl;
                std::cout << "      Total slice size: " << total_slice_size
                          << std::endl;
            }

            auto kernel = &sclx_kernel<
                std::decay_t<F>,
                std::decay_t<IndexGenerator>,
                ThreadBlockRank>;

            uint forty_eight_kb = 1024 * 48;
            uint maxbytes
                = std::max(forty_eight_kb, static_cast<uint>(local_mem_size));
            auto raw_error = cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxbytes
            );
            sclx::cuda::cuda_exception::raise_if_not_success(
                raw_error,
                std::experimental::source_location(),
                "sclx::detail::"
            );

            kernel<<<
                grid_size,
                block_shape.elements(),
                local_mem_size,
                stream>>>(f, metadata, index_generator, slice_idx, slice_size);
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

}  // namespace sclx::detail

#define REGISTER_SCALIX_KERNEL_TAG(Tag)                                        \
    template<class F, uint RangeRank, uint ThreadBlockRank>                    \
    __global__ void sclx_##Tag(                                                \
        F f,                                                                   \
        sclx::kernel_info<RangeRank, ThreadBlockRank> metadata                 \
    ) {                                                                        \
        const sclx::md_range_t<RangeRank>& global_range                        \
            = metadata.global_range();                                         \
        sclx::md_index_t<RangeRank> idx = metadata.global_thread_id();         \
        while (idx.as_linear(global_range)                                     \
               < metadata.start_index().as_linear(global_range)                \
                     + metadata.device_range().elements()) {                   \
            f(idx, metadata);                                                  \
                                                                               \
            idx = sclx::md_index_t<RangeRank>::create_from_linear(             \
                idx.as_linear(global_range) + metadata.grid_stride(),          \
                global_range                                                   \
            );                                                                 \
            metadata.increment_stride_count();                                 \
        }                                                                      \
    }                                                                          \
                                                                               \
    template<class F, class IndexGenerator, uint ThreadBlockRank>              \
    __global__ void sclx_##Tag(                                                \
        F f,                                                                   \
        sclx::kernel_info<IndexGenerator::range_rank, ThreadBlockRank>         \
            metadata,                                                          \
        IndexGenerator idx_gen,                                                \
        sclx::index_t slice_idx,                                               \
        sclx::index_t slice_size                                               \
    ) {                                                                        \
        constexpr uint range_rank = IndexGenerator::range_rank;                \
        constexpr uint index_rank = IndexGenerator::index_rank;                \
        const sclx::md_range_t<range_rank>& global_range                       \
            = metadata.global_range();                                         \
        sclx::md_index_t<range_rank> global_thread_id                          \
            = metadata.global_thread_id();                                     \
                                                                               \
        while (global_thread_id.as_linear(global_range)                        \
               < global_range.elements()) {                                    \
            sclx::md_index_t<index_rank> idx = idx_gen(global_thread_id);      \
                                                                               \
            if (idx[index_rank - 1] >= slice_idx                               \
                && idx[index_rank - 1] < slice_idx + slice_size) {             \
                f(idx, metadata);                                              \
            }                                                                  \
                                                                               \
            global_thread_id = sclx::md_index_t<>::create_from_linear(         \
                global_thread_id.as_linear(global_range)                       \
                    + metadata.grid_stride(),                                  \
                global_range                                                   \
            );                                                                 \
            metadata.increment_stride_count();                                 \
        }                                                                      \
    }                                                                          \
                                                                               \
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
                sclx::cuda::set_device(device_id); /*init context*/            \
                                                                               \
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
                sclx::kernel_info<RangeRank, ThreadBlockRank> metadata(        \
                    global_range,                                              \
                    device_range,                                              \
                    block_shape,                                               \
                    device_start_idx,                                          \
                    iter                                                       \
                );                                                             \
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
                                                                               \
                auto kernel = &sclx_##Tag<                                     \
                    std::decay_t<F>,                                           \
                    RangeRank,                                                 \
                    ThreadBlockRank>;                                          \
                                                                               \
                uint forty_eight_kb = 1024 * 48;                               \
                uint maxbytes       = std::max(                                \
                    forty_eight_kb,                                      \
                    static_cast<uint>(local_mem_size)                    \
                );                                                       \
                auto raw_error = cudaFuncSetAttribute(                         \
                    kernel,                                                    \
                    cudaFuncAttributeMaxDynamicSharedMemorySize,               \
                    maxbytes                                                   \
                );                                                             \
                sclx::cuda::cuda_exception::raise_if_not_success(              \
                    raw_error,                                                 \
                    std::experimental::source_location(),                      \
                    "sclx::detail::"                                           \
                );                                                             \
                                                                               \
                kernel<<<                                                      \
                    grid_size,                                                 \
                    block_shape.elements(),                                    \
                    local_mem_size,                                            \
                    stream>>>(f, metadata);                                    \
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
                                                                               \
        template<class IndexGenerator, uint ThreadBlockRank, class F>          \
        __host__ static void execute(                                          \
            const std::vector<std::tuple<int, size_t, size_t>>& device_info,   \
            IndexGenerator&& index_generator,                                  \
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
            size_t total_slice_size = 0;                                       \
            if constexpr (sclx::detail::_debug_kernel_launch) {                \
                sclx::detail::_debug_kernel_mutex::mutex.lock();               \
                std::cout << "Launching Scalix Kernel using IndexGenerator"    \
                          << std::endl;                                        \
                for (auto& [device_id, slice_idx, slice_size] : device_info) { \
                    total_slice_size += slice_size;                            \
                }                                                              \
            }                                                                  \
                                                                               \
            int iter = 0;                                                      \
            for (auto& [device_id, slice_idx, slice_size] : device_info) {     \
                sclx::cuda::set_device(device_id); /* init context */          \
                                                                               \
                size_t total_threads = index_generator.range().elements();     \
                size_t max_grid_size                                           \
                    = (total_threads + block_shape.elements() - 1)             \
                    / block_shape.elements();                                  \
                grid_size = std::min(max_grid_size, grid_size);                \
                                                                               \
                constexpr uint range_rank                                      \
                    = std::remove_reference_t<IndexGenerator>::range_rank;     \
                sclx::kernel_info<range_rank, ThreadBlockRank> metadata(       \
                    index_generator.range(),                                   \
                    index_generator.range(),                                   \
                    block_shape,                                               \
                    {},                                                        \
                    iter                                                       \
                );                                                             \
                                                                               \
                auto& stream = streams[iter++];                                \
                                                                               \
                if constexpr (sclx::detail::_debug_kernel_launch) {            \
                    std::cout << "  Launching kernel on device " << device_id  \
                              << std::endl;                                    \
                    std::cout << "      Device slice size: " << slice_size     \
                              << std::endl;                                    \
                    std::cout                                                  \
                        << "      Device slice start index: " << slice_idx     \
                        << std::endl;                                          \
                    std::cout                                                  \
                        << "      Total slice size: " << total_slice_size      \
                        << std::endl;                                          \
                }                                                              \
                                                                               \
                auto kernel = &sclx_##Tag<                                     \
                    std::decay_t<F>,                                           \
                    std::decay_t<IndexGenerator>,                              \
                    ThreadBlockRank>;                                          \
                                                                               \
                uint forty_eight_kb = 1024 * 48;                               \
                uint maxbytes       = std::max(                                \
                    forty_eight_kb,                                      \
                    static_cast<uint>(local_mem_size)                    \
                );                                                       \
                auto raw_error = cudaFuncSetAttribute(                         \
                    kernel,                                                    \
                    cudaFuncAttributeMaxDynamicSharedMemorySize,               \
                    maxbytes                                                   \
                );                                                             \
                sclx::cuda::cuda_exception::raise_if_not_success(              \
                    raw_error,                                                 \
                    std::experimental::source_location(),                      \
                    "sclx::detail::"                                           \
                );                                                             \
                                                                               \
                kernel<<<                                                      \
                    grid_size,                                                 \
                    block_shape.elements(),                                    \
                    local_mem_size,                                            \
                    stream>>>(                                                 \
                    f,                                                         \
                    metadata,                                                  \
                    index_generator,                                           \
                    slice_idx,                                                 \
                    slice_size                                                 \
                );                                                             \
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
