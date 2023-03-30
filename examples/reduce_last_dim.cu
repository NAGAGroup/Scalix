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
 * @file reduce_last_dim.cu
 * @brief Example of using reduce_last_dim
 *
 * Often times, it is useful to reduce along the last dimension of an array.
 * For example, if we wanted to find the lower bound of a bounding box on
 * a set of 3D points stored in a sclx::array<float, 2> with shape (3, N)
 *
 * This algorithm scales poorly unless the reduction functor is computationally
 * expensive. However, it avoids expensive memory transfers by remaining
 * distributed. If you don't want to distribute the work, first set the
 * preferred devices to a single device.
 *
 * In this example we show how distributed vs. single device performance
 * compares. Single device performance including and excluding memory transfers
 * are both shown. Of course, if only one device is available, the distributed
 * and single device performance will be the same.
 */

#include <chrono>
#include <scalix/algorithm/reduce_last_dim.cuh>

int main() {
    sclx::array<uint, 4> a{3, 3, 3, 5'000'000};
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<4>{a.shape()},
            a,
            [=] __device__(const sclx::md_index_t<4>& index, const auto&) {
                a[index] = index[3];
            }
        );
    }).get();

    uint identity = std::numeric_limits<uint>::min();

    auto now     = std::chrono::high_resolution_clock::now();
    auto reduced = sclx::algorithm::reduce_last_dim(
        a,
        identity,
        sclx::algorithm::max<>()
    );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - now;
    std::cout << "Time for distributed reduce: " << elapsed.count() / 1000
              << "ms" << std::endl;

    reduced.prefetch_async({sclx::cuda::traits::cpu_device_id});
    for (auto& val : reduced) {
        if (val != 4'999'999) {
            std::cerr << "Error: " << val << std::endl;
        }
    }

    // let's do the same, but this time moving all the work to a single device
    now = std::chrono::high_resolution_clock::now();
    a.set_primary_devices(std::vector<int>{sclx::cuda::traits::current_device()}
    );

    auto now_after_mem_transfer = std::chrono::high_resolution_clock::now();
    reduced                     = sclx::algorithm::reduce_last_dim(
        a,
        identity,
        sclx::algorithm::max<>()
    );
    end     = std::chrono::high_resolution_clock::now();
    elapsed = end - now;
    std::cout << "Time for single device reduce, including memory transfers: "
              << elapsed.count() / 1000 << "ms" << std::endl;
    elapsed = end - now_after_mem_transfer;
    std::cout << "Time for single device reduce, excluding memory transfers: "
              << elapsed.count() / 1000 << "ms" << std::endl;

    reduced.prefetch_async({sclx::cuda::traits::cpu_device_id});
    for (auto& val : reduced) {
        if (val != 4'999'999) {
            std::cerr << "Error: " << val << std::endl;
        }
    }

    return 0;
}
