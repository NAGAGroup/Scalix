
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
#include "range.cuh"
#include "shape.cuh"
#include "index.cuh"

namespace sclx {
template<uint ProblemRank, uint ThreadBlockRank>
class kernel_info {
  public:
    kernel_info(const kernel_info&)                = default;
    kernel_info(kernel_info&&) noexcept            = default;
    kernel_info& operator=(const kernel_info&)     = default;
    kernel_info& operator=(kernel_info&&) noexcept = default;

    __host__ kernel_info(
        const md_range_t<ProblemRank>& global_range,
        const md_range_t<ProblemRank>& device_range,
        const shape_t<ThreadBlockRank>& thread_block_shape,
        const md_index_t<ProblemRank>& start_index,
        const int& device_id
    )
        : thread_block_shape_(thread_block_shape),
          global_range_(global_range),
          device_range_(device_range),
          start_index_(start_index),
          device_id_(device_id) {}

    __host__ __device__ const shape_t<ThreadBlockRank>&
    thread_block_shape() const {
        return thread_block_shape_;
    }

    __host__ __device__ const md_range_t<ProblemRank>& global_range() const {
        return global_range_;
    }

    __host__ __device__ const md_range_t<ProblemRank>& device_range() const {
        return device_range_;
    }

    __host__ __device__ const md_index_t<ProblemRank>& start_index() const {
        return start_index_;
    }

    __device__ md_index_t<ThreadBlockRank> local_thread_id() const {
        return md_index_t<>::create_from_linear(
            threadIdx.x,
            thread_block_shape_
        );
    }

    __device__ md_index_t<ProblemRank> global_thread_id() const {
        return md_index_t<>::create_from_linear(
            blockIdx.x * blockDim.x + threadIdx.x
                + start_index_.as_linear(global_range_),
            global_range_
        );
    }

    __host__ __device__ const int& device_id() const { return device_id_; }

  private:
    shape_t<ThreadBlockRank> thread_block_shape_;
    md_range_t<ProblemRank> global_range_;
    md_range_t<ProblemRank> device_range_;
    md_index_t<ProblemRank> start_index_;
    int device_id_;
};
}
