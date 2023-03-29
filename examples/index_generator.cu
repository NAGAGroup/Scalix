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

/** \file index_generator.cu
 *  \brief Run kernels using a multidimensional index generator.
 *
 *  Sometimes, the write pattern to a result array is not known at compile time.
 *
 *  Usually this occurs when we query some data structure for an index to write
 *  to. For example, imagine we want to create a distributed histogram for
 *  an array of values for which we know nothing about the distribution. To
 *  solve this problem sequentially, we would iterate over the array of values,
 *  determine the bin for each value, and increment the bin count for that bin.
 *
 *  Doing this on a distributed system is a bit more complicated. To this
 *  with the typical range-based kernel API would be impossible, as we don't
 *  know anything about the structure of the values and therefore have no way
 *  of distributing the work so that writes are local to each device.
 *
 *  Instead, we can use the index generator kernel API. An index generator
 *  transforms a thread index into a write index. Internally, instead of
 *  distributing the work across the devices, we replicate the work across all
 *  devices, running the total thread grid on each device. Before calling
 *  the device functor with the index returned by the generator, we check if
 *  the write index is valid for the executing device. If it is, we call the
 *  device functor with the index. If it is not, we do nothing. The index
 *  generator in our histogram example would be responsible for converting a
 *  thread index, which corresponds to a value in the array, into a bin index.
 *
 *  While this likely won't show much performance improvement over running
 *  the kernel on a single device, it does instead prevent expensive data
 *  transfers between devices. This is useful when most of an algorithm is
 *  ideal for distributed execution, but a small part of the algorithm is
 *  not.
 */

#include <scalix/array.cuh>
#include <scalix/execute_kernel.cuh>
#include <scalix/fill.cuh>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

template<uint IndexRank, uint RangeRank>
sclx::array<sclx::md_index_t<IndexRank>, RangeRank> generate_random_indices(
    const sclx::shape_t<RangeRank>& generator_shape,
    const sclx::shape_t<IndexRank>& target_shape,
    int seed = 0
) {
    sclx::array<sclx::md_index_t<IndexRank>, RangeRank> indices(generator_shape
    );

    // we don't have a distributed algorithm for generating random numbers
    // so we move the array to the current device and use the thrust API
    indices.unset_read_mostly();
    indices.prefetch_async({sclx::cuda::traits::current_device()});
    thrust::counting_iterator<size_t> first(0);
    thrust::transform(
        thrust::device,
        first,
        first + indices.elements(),
        indices.begin(),
        [=] __host__ __device__(size_t n) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<int> dist(
                0,
                target_shape.elements() - 1
            );
            rng.discard(n);

            auto linear_index = dist(rng);
            return sclx::md_index_t<IndexRank>::create_from_linear(
                linear_index,
                target_shape
            );
        }
    );
    indices.set_read_mostly();
    indices.prefetch_async(sclx::exec_topology::distributed);

    return indices;
}

template<uint IndexRank, uint RangeRank = 1>
class random_index_generator {
  public:
    static constexpr uint range_rank
        = RangeRank;  ///< Used to generate the thread grid with indices of type
                      ///< md_index_t<Rank> (required by execute_kernel).
    static constexpr uint index_rank
        = IndexRank;  ///< Used to generate the write indices with indices of
                      ///< type md_index_t<Rank> (required by execute_kernel).

    random_index_generator(
        const sclx::shape_t<IndexRank>& target_shape,
        const sclx::shape_t<RangeRank>& generator_shape,
        int seed = 0
    )
        : generator_shape_(generator_shape),
          target_shape_(target_shape) {
        indices_ = generate_random_indices(generator_shape, target_shape, seed);
    }

    __host__ __device__ const sclx::md_index_t<IndexRank>&
    operator()(sclx::md_index_t<RangeRank> index) const {
        return indices_[index];
    }

    __host__ __device__ const sclx::md_range_t<RangeRank>& range() const {
        return static_cast<const sclx::md_range_t<RangeRank>&>(generator_shape_
        );
    }

  private:
    sclx::shape_t<RangeRank> generator_shape_;
    sclx::shape_t<IndexRank> target_shape_;
    sclx::array<sclx::md_index_t<IndexRank>, RangeRank> indices_;
};

int main() {
    sclx::array<int, 3> arr({4, 4, 4});
    sclx::fill(arr, 0);
    sclx::shape_t<2> generator_shape({100, 100});
    random_index_generator<3, 2> generator(arr.shape(), generator_shape);

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            generator,
            arr,
            [=] __device__(const sclx::md_index_t<3>& index) {
                atomicAdd(&arr[index], 1);
            }
        );
    });

    arr.prefetch_async({sclx::cuda::traits::cpu_device_id});
    for (size_t linear_idx = 0; linear_idx < arr.elements(); ++linear_idx) {
        auto index
            = sclx::md_index_t<3>::create_from_linear(linear_idx, arr.shape());
        std::cout << "arr[" << index << "] = " << arr[index] << std::endl;
    }
}
