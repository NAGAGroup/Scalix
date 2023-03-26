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

/** @file inclusive_scan.cu
 * @brief This example shows how to use the inclusive_scan algorithm
 */

#include <chrono>
#include <scalix/algorithm/inclusive_scan.cuh>

int main() {
    // We use a multi-dimensional array to store the data
    // to show how the inclusive_scan algorithm works with
    // multi-dimensional arrays. It essentially flattens
    // the data and then performs the inclusive scan on
    // the flattened data.
    //
    // Why even allow multi-dimensional arrays? Once can imagine
    // some partitioning data structure that partitions point cloud
    // data. Each of these partitions could store a different,
    // unknown at runtime, number of points. We may have a single
    // array of indices and each partition has a pointer to a region
    // in this array. So we'd first count the number of points in each
    // partition, then do an inclusive scan on the counts to get the
    // starting index of each partition. For readability, we'd likely
    // want to use a multi-dimensional array to store the counts so that
    // it matches the multi-dimensional array of partitions.
    sclx::array<int, 3> arr{2, 2, 1'000'000};

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<3>{arr.shape()},
            arr,
            [=] __device__(const sclx::md_index_t<3>& idx) { arr[idx] = 1; }
        );
    }).get();

    // this first example, we use a simple plus<> functor to show correctness
    // of the algorithm
    // it won't show any weak or strong scaling benefits since the GPUs
    // are not computationally or memory bound
    auto start = std::chrono::high_resolution_clock::now();
    auto result
        = sclx::algorithm::inclusive_scan(arr, 0, sclx::algorithm::plus<>());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - start
                 )
                         .count()
                     / 1000.0
              << "ms" << std::endl;

    arr.prefetch_async({sclx::cuda::traits::cpu_device_id});

    for (int i = 0; i < result.elements(); i++) {
        if (result.data().get()[i] != i + 1) {
            std::cout << "Error at index " << i << ": "
                      << result.data().get()[i] << " != " << i + 1 << std::endl;
        }
    }

    // now that we know the algorithm works, we can use a more computationally
    // expensive functor to show the benefits of weak and strong scaling
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<3>{arr.shape()},
            arr,
            [=] __device__(const sclx::md_index_t<3>& idx) { arr[idx] = 1; }
        );
    }).get();

    start  = std::chrono::high_resolution_clock::now();
    result = sclx::algorithm::inclusive_scan(
        arr,
        0,
        [] __host__ __device__(int a, int b) {
            for (int i = 0; i < 100'000; ++i) {
                a *= static_cast<int>(sqrtf(static_cast<float>(b)));
            }
            return a;
        }
    );
    end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - start
                 )
                         .count()
                     / 1000.0
              << "ms" << std::endl;

    return 0;
}
