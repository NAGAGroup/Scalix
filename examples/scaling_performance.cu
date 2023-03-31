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

/**
 * @file scaling_performance.cu
 * @brief Example of how a computationally intensive kernel scales with devices
 *
 * This example simulates a computationally intensive kernel with a consistent,
 * but unordered read pattern. We run the kernel twice to show the benefits
 * of the CUDA read-mostly hint that Scalix uses under the hood that allows
 * read-only copies of data to be distributed across devices concurrently.
 *
 * When running more than one device, we see the first call to the kernel
 * take longer than the second, as the on-demand page migration has not yet
 * made the aforementioned read-only copies of the data.
 */

#include <chrono>
#include <scalix/scalix.cuh>
#include <scalix/fill.cuh>
#include "utilities/random_index_generator.cuh"

int main() {
    sclx::array<float, 3> source{4, 4, 5'000'000};
    sclx::fill(source, 2.0f);
    sclx::array<float, 3> target{4, 4, 5'000'000};

    random_index_generator<3, 3> generator(source.shape(), target.shape());

    auto start = std::chrono::high_resolution_clock::now();
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        sclx::local_array<float, 1> local_cache(handler, sclx::cuda::traits::kernel::default_block_shape);
        handler.launch(
            sclx::md_range_t<3>{target.shape()},
            target,
            [=] __device__(const sclx::md_index_t<3>& index, const auto& info) mutable {
                auto local_thread_id = info.local_thread_id();
                local_cache[local_thread_id] = source[generator(index)];
                for (int i = 0; i < 1000; ++i) {
                    target[index] = sqrt(local_cache[local_thread_id]);
                }
            }
        );
    }).get();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() * 1000 << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        sclx::local_array<float, 1> local_cache(handler, sclx::cuda::traits::kernel::default_block_shape);
        handler.launch(
            sclx::md_range_t<3>{target.shape()},
            target,
            [=] __device__(const sclx::md_index_t<3>& index, const auto& info) mutable {
                auto local_thread_id = info.local_thread_id();
                local_cache[local_thread_id] = source[generator(index)];
                for (int i = 0; i < 1000; ++i) {
                    target[index] = sqrt(local_cache[local_thread_id]);
                }
            }
        );
    }).get();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() * 1000 << " ms" << std::endl;
}
