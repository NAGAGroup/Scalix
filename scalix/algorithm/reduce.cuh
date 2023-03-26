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
#include "../array.cuh"
#include "../execute_kernel.cuh"
#include "functional.cuh"
#include <mutex>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace sclx::algorithm {

template<class T, uint Rank, class F>
__host__ T reduce(const array<T, Rank>& arr, const T& identity, F&& f) {
    const auto& mem_info = arr.memory_info();
    std::vector<int> devices(
        mem_info.devices.get(),
        mem_info.devices.get() + mem_info.num_devices
    );

    std::vector<std::future<void>> futures;
    futures.reserve(devices.size());

    std::mutex mutex;

    int current_device = cuda::traits::current_device();

    auto device_info = get_device_split_info(arr);

    T result = identity;

#ifndef __CLION_IDE__  // CLion shows incorrect errors with thrust library
    for (auto& [_device_id, _slice_idx, _slice_size] : device_info) {

        auto lambda = [&](int device_id, size_t slice_idx, size_t slice_size) {
            cuda::set_device(device_id);  // init context for thread

            auto arr_slice
                = arr.get_range({slice_idx}, {slice_idx + slice_size});

            auto stream = cuda::stream_t::create_for_device(device_id);
            auto tmp    = thrust::reduce(
                thrust::cuda::par.on(stream),
                arr_slice.begin(),
                arr_slice.end(),
                identity,
                f
            );

            std::lock_guard<std::mutex> lock(mutex);
            result = f(result, tmp);
        };

        futures.emplace_back(std::async(
            std::launch::async,
            lambda,
            _device_id,
            _slice_idx,
            _slice_size
        ));
    }

    for (auto& fut : futures) {
        fut.get();
    }

    cuda::set_device(current_device);
#endif
    return result;
}

}  // namespace sclx::algorithm