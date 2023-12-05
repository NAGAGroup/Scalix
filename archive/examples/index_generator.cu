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

#include "utilities/random_index_generator.cuh"
#include <scalix/array.cuh>
#include <scalix/execute_kernel.cuh>
#include <scalix/fill.cuh>

REGISTER_SCALIX_KERNEL_TAG(random_index_generator_example);

int main() {
    sclx::array<int, 3> arr({4, 4, 4});
    sclx::fill(arr, 0);
    sclx::shape_t<2> generator_shape({100, 100});
    random_index_generator<3, 2> generator(arr.shape(), generator_shape);

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch<random_index_generator_example>(
            generator,
            arr,
            [=] __device__(const sclx::md_index_t<3>& index, const auto&) {
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
