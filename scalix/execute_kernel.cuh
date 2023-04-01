#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"
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
#include "array_list.cuh"
#include "array_tuple.cuh"
#include "cuda.hpp"
#include "detail/execute_kernel.cuh"
#include <future>
#include <mutex>

namespace sclx {

class kernel_handler {
  public:
    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        class T,
        uint ResultRank,
        uint NResult,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank()>
    __host__ void launch(
        md_range_t<RangeRank> range,
        array_list<T, ResultRank, NResult> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch_array_list_impl<KernelTag>(
            range,
            result,
            block_shape,
            grid_size,
            std::forward<F>(f)
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        class T,
        uint ResultRank,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank()>
    __host__ void launch(
        md_range_t<RangeRank> range,
        dynamic_array_list<T, ResultRank> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch_array_list_impl<KernelTag>(
            range,
            result,
            block_shape,
            grid_size,
            std::forward<F>(f)
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        class T,
        uint ResultRank,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank()>
    __host__ void launch(
        md_range_t<RangeRank> range,
        array<T, ResultRank> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch(
            range,
            array_list<T, ResultRank, 1>({result}),
            std::forward<F>(f),
            block_shape,
            grid_size
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        class IndexGenerator,
        class T,
        uint ResultRank,
        uint NResult,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank()>
    __host__ void launch(
        IndexGenerator&& index_generator,
        array_list<T, ResultRank, NResult> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch_array_list_impl<KernelTag>(
            std::forward<IndexGenerator>(index_generator),
            result,
            block_shape,
            grid_size,
            std::forward<F>(f)
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        class IndexGenerator,
        class T,
        uint ResultRank,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank()>
    __host__ void launch(
        IndexGenerator&& index_generator,
        dynamic_array_list<T, ResultRank> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch_array_list_impl<KernelTag>(
            std::forward<IndexGenerator>(index_generator),
            result,
            block_shape,
            grid_size,
            std::forward<F>(f)
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        class IndexGenerator,
        class T,
        uint ResultRank,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank()>
    __host__ void launch(
        IndexGenerator&& index_generator,
        array<T, ResultRank> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch(
            index_generator,
            array_list<T, ResultRank, 1>({result}),
            std::forward<F>(f),
            block_shape,
            grid_size
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank(),
        class... ArrayTypes>
    __host__ void launch(
        md_range_t<RangeRank> range,
        array_tuple<ArrayTypes...> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch_tuple_impl<KernelTag>(
            range,
            result,
            block_shape,
            grid_size,
            std::forward<F>(f)
        );
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        class IndexGenerator,
        class F,
        uint ThreadBlockRank = cuda::traits::kernel::default_block_shape.rank(),
        class... ArrayTypes>
    __host__ void launch(
        IndexGenerator&& index_generator,
        array_tuple<ArrayTypes...> result,
        F&& f,
        shape_t<ThreadBlockRank> block_shape
        = cuda::traits::kernel::default_block_shape,
        size_t grid_size = cuda::traits::kernel::default_grid_size
    ) const {
        launch_tuple_impl<KernelTag>(
            index_generator,
            result,
            block_shape,
            grid_size,
            std::forward<F>(f)
        );
    }

    __device__ void syncthreads() const { __syncthreads(); }
    __device__ void threadfence() const { __threadfence(); }

    template<class T, uint Rank>
    friend class local_array;

  private:
    __device__ static char* get_local_mem(const size_t& offset) {
        extern __shared__ char local_mem[];
        return local_mem + offset;
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        uint ThreadBlockRank,
        class ArrayListType,
        class F>
    __host__ void launch_array_list_impl(
        md_range_t<RangeRank> range,
        ArrayListType&& result,
        shape_t<ThreadBlockRank> block_shape,
        size_t grid_size,
        F&& f
    ) const {
        auto device_split_first = get_device_split_info(result[0]);
        size_t last_dim         = result[0].shape()[result[0].rank() - 1];
        for (auto& array : result) {
            if (!is_same_device_split(
                    device_split_first,
                    get_device_split_info(array)
                )) {
                array.set_primary_devices(device_split_first);
                std::cerr
                    << "Warning: Not every array in the list has the same "
                       "memory split, setting primary devices for all "
                       "arrays to the same as the first array in the list"
                    << std::endl;
            }
            if (array.shape()[array.rank() - 1] != last_dim) {
                throw_exception<std::invalid_argument>("All arrays in the result must "
                                            "have the same last dimension");
            }
            array.unset_read_mostly();
        }

        auto device_info = get_device_split_info(result[0]);

        KernelTag::template execute<RangeRank, ThreadBlockRank>(
            device_info,
            range,
            block_shape,
            grid_size,
            local_mem_size_,
            std::forward<F>(f)
        );

        for (auto& array : result) {
            array.set_read_mostly();
        }
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        class IndexGenerator,
        uint ThreadBlockRank,
        class ArrayListType,
        class F>
    __host__ void launch_array_list_impl(
        IndexGenerator&& index_generator,
        ArrayListType&& result,
        shape_t<ThreadBlockRank> block_shape,
        size_t grid_size,
        F&& f
    ) const {
        auto device_split_first = get_device_split_info(result[0]);
        for (auto& array : result) {
            if (!is_same_device_split(
                    device_split_first,
                    get_device_split_info(array)
                )) {
                array.set_primary_devices(device_split_first);
                std::cerr
                    << "Warning: Not every array in the list has the same "
                       "memory split, setting primary devices for all "
                       "arrays to the same as the first array in the list"
                    << std::endl;
            }

            using generator_t = std::remove_reference_t<IndexGenerator>;
            if (index_generator.index_range()[generator_t::index_rank - 1]
                != array.shape()[array.rank() - 1]) {
                throw std::invalid_argument("Index generator indices and "
                                            "result array must have the same "
                                            "last dimension");
            }

            array.unset_read_mostly();
        }

        auto device_info = get_device_split_info(result[0]);

        KernelTag::template execute<IndexGenerator, ThreadBlockRank>(
            device_info,
            index_generator,
            block_shape,
            grid_size,
            local_mem_size_,
            std::forward<F>(f)
        );

        for (auto& array : result) {
            array.set_read_mostly();
        }
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        uint RangeRank,
        uint ThreadBlockRank,
        class F,
        class... ArrayTypes>
    __host__ void launch_tuple_impl(
        md_range_t<RangeRank> range,
        array_tuple<ArrayTypes...> result,
        shape_t<ThreadBlockRank> block_shape,
        size_t grid_size,
        F&& f
    ) const {
        if (!result.last_dims_all_equal()) {
            throw_exception<std::invalid_argument>("All arrays in the result must "
                                                   "have the same last dimension");
        }

        result.unset_read_mostly();

        auto device_info = get_device_split_info(
            thrust::get<0>(static_cast<thrust::tuple<ArrayTypes...>>(result))
        );

        KernelTag::template execute<RangeRank, ThreadBlockRank>(
            device_info,
            range,
            block_shape,
            grid_size,
            local_mem_size_,
            std::forward<F>(f)
        );

        result.set_read_mostly();
    }

    template<
        class KernelTag = detail::default_kernel_tag,
        class IndexGenerator,
        uint ThreadBlockRank,
        class F,
        class... ArrayTypes>
    __host__ void launch_tuple_impl(
        IndexGenerator&& index_generator,
        array_tuple<ArrayTypes...> result,
        shape_t<ThreadBlockRank> block_shape,
        size_t grid_size,
        F&& f
    ) const {
        using generator_t = std::remove_reference_t<IndexGenerator>;
        if (index_generator.index_range()[generator_t::index_rank - 1]
            != thrust::get<0>(static_cast<thrust::tuple<ArrayTypes...>>(result))
                   .shape()[0]) {
            throw_exception<std::invalid_argument>("Index generator indices and "
                                        "result array must have the same "
                                        "last dimension");
        }
        if (!result.last_dims_all_equal()) {
            throw_exception<std::invalid_argument>("All arrays in the result must "
                                                   "have the same last dimension");
        }

        result.unset_read_mostly();

        auto device_info = get_device_split_info(
            thrust::get<0>(static_cast<thrust::tuple<ArrayTypes...>>(result))
        );

        KernelTag::template execute<IndexGenerator, ThreadBlockRank>(
            device_info,
            index_generator,
            block_shape,
            grid_size,
            local_mem_size_,
            std::forward<F>(f)
        );

        result.set_read_mostly();
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
        return data_[idx.as_linear(shape_)];
    }

    __device__ T& operator[](const index_t& idx) {
        if (!data_) {
            data_
                = reinterpret_cast<T*>(kernel_handler::get_local_mem(offset_));
        }
        return data_[idx];
    }

    template<class... Args>
    __device__ T& operator()(const Args&... args) {
        return operator[](md_index_t<Rank>{static_cast<const size_t&>(args)...}
        );
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

#pragma clang diagnostic pop