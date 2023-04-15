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

/** \file distributed_indexing.cu
 * \brief This example showcases read indexing using a random access pattern
 *
 * We show that reading from one device to another still scales well due to
 * the use of unified memory with on-demand page migration and proper access
 * hints (which Scalix provides under the hood). The random access pattern
 * was checked to ensure that reads were actually being made across device
 * boundaries.
 *
 * For fun, we also show the distributed reduce algorithm in action.
 */

#include <scalix/algorithm/reduce.cuh>
#include <scalix/scalix.cuh>
#include <chrono>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

// this tag will be used to name a cuda kernel for any
// kernel launch that uses this tag
// useful for profiling
REGISTER_SCALIX_KERNEL_TAG(my_kernel_tag);

int main() {

    // create 2 1D arrays
    //
    // the arrays are automatically initialized such that their memory
    // is evenly distributed across all available devices
    //
    // since we are using unified memory under the hood, we can access
    // the data from any device, but the hints provided to CUDA are such that
    // reads/writes are performed best on the device that "owns" the memory
    //
    // further, we have set the read-mostly flag on the arrays by default. while
    // counter-intuitive, we have found that this improves performance for
    // arrays that are mostly accessed by the device that "owns" the memory even
    // if that access tends to be mostly writes. if you know that a particular
    // code block is going to have an access pattern that causes writes to be
    // executed non-locally (i.e. on a device other than the one that "owns" the
    // memory), it is best to temporarily unset the read-mostly flag on the
    // array and then set it back to true once the code block has completed (see
    // the random integer generation code below)
    size_t numelem = 100'000;
    sclx::array<float, 1> arr2({numelem});
    sclx::array<float, 1> arr3({numelem});

    // here we create an indexing array where the last dimension is the same
    // as the above two arrays
    //
    // this is because our problem will alternate between reading 64 elements
    // from arr2 and adding them to arr3 and vice versa. the 64 elements are
    // randomized and stored in the indexing array
    sclx::array<size_t, 2> indices({64, numelem});

    // fill indices with random numbers
    indices.unset_read_mostly();
    indices.prefetch_async({sclx::cuda::traits::current_device()});
    thrust::counting_iterator<size_t> begin(0);
    thrust::transform(
        thrust::device,
        begin,
        begin + numelem * 64,
        indices.begin(),
        [=] __host__ __device__(size_t n) {
            auto rng = thrust::default_random_engine{0};
            thrust::uniform_int_distribution<size_t> dist(0, numelem - 1);
            rng.discard(n);
            return dist(rng);
        }
    );
    indices.set_read_mostly();

    // the indices are then re-fetched to their owning devices in a
    // distributed fashion, as opposed to replicated (which is the default)
    indices.prefetch_async(sclx::exec_topology::distributed);

    // here we launch the kernel, which returns a future that we immediately
    // call .get() on to block until the kernel has completed and check
    // for any errors
    //
    // there are two stages to this process:
    // 1. we call execute_kernel and pass our host lambda
    //    this lambda allows for some staging to be done host-side like allocate
    //    shared memory
    // 2. we call handler.launch and pass the iteration range, the array that
    //    is to be our result from the kernel, and the device lambda. we require
    //    a result array because it tells the backend how to split the problem
    //    across devices. we also support an array_list of result arrays, for
    //    which they must all have the same last dimension as this is internally
    //    how memory is split across devices
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            /* iteration range */ sclx::md_range_t<1>{arr3.shape()},
            /* result array */ arr3,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                arr3[idx] = 1.f;
            }
        );
    }).get();

    auto now = std::chrono::high_resolution_clock::now(
    );  // timer for checking 1 vs. 2 devices

    // now we immediately switch the read/write arrays to show the power of
    // CUDA's unified memory, specifically on-demand page migration
    //
    // because of on-demand page migration, the previously invalidated read-only
    // copies of the data only need to be updated for pages accessed. no need
    // to call prefetch as this actually hurts performance for Scalix's
    // distributed architecture
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<2>{indices.shape()},
            arr2,
            [=] __device__(const sclx::md_index_t<2>& idx, const auto&) {
                atomicAdd(&arr2(idx[1]), arr3(indices[idx]));
            }
        );
    }).get();

    std::cout << "arr2(0): " << arr2(0) << std::endl;
    std::cout << "arr3(0): " << arr3(0) << std::endl;

    // in this following example, we show how one would use cuda shared memory
    // although not really necessary for this example
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        auto shared_array = sclx::local_array<sclx::md_index_t<2>, 1>(
            handler,
            sclx::cuda::traits::kernel::default_block_shape
        );

        handler.launch(
            sclx::md_range_t<2>{indices.shape()},
            arr3,
            [=] __device__(
                const sclx::md_index_t<2>& idx,
                const auto& info
            ) mutable {  // mutable is required for local_array
                auto local_index          = info.local_thread_id();
                shared_array[local_index] = idx;
                handler.syncthreads();

                atomicAdd(
                    &arr3(shared_array[local_index][1]),
                    arr2(indices[shared_array[local_index]])
                );
            }
        );
    }).get();

    std::cout << "arr2(0): " << arr2(0) << std::endl;
    std::cout << "arr3(0): " << arr3(0) << std::endl;

    // in this example we show how to set up a different thread grid shape
    // than the default
    //
    // note that internally we use a strided grid step to ensure the entire
    // problem is executed, even if the total thread is not >= the problem size
    //
    // this allows the user to play around with different thread grid shapes
    // to see how it affects performance
    //
    // we also use a 1D iteration range, so we don't need to do atomicAdd
    // we showcase array slicing
    //
    // finally, we also tag the kernel for easier profiling/debugging
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch<my_kernel_tag>(  // tag is used to identify the kernel
            /* iteration range */ sclx::md_range_t<1>{arr2.shape()},
            /* result array */ arr2,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                // we can slice arrays in both host and device code
                auto index_slice = indices.get_slice(idx);

                for (auto& read_idx : index_slice) {
                    arr2[idx] += arr3(read_idx);
                }
            },
            /* thread block shape */ sclx::shape_t<1>{1},
            /* grid size */ 1  // note that this needs to be a scalar
        );
    }).get();

    std::cout << "arr2(0): " << arr2(0) << std::endl;
    std::cout << "arr3(0): " << arr3(0) << std::endl;

    // here we show our reduction algorithm, partially just for fun, but also
    // because it is checks that the solution is correct (expected value is
    // 4096 * numelem)
    auto sum_of_arr3
        = sclx::algorithm::reduce(arr3, 0.f, sclx::algorithm::plus<>());

    std::cout << "sum of arr3: " << sum_of_arr3 << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - now
                 )
                         .count()
                     / 1000.f
              << "ms" << std::endl;

    return 0;
}