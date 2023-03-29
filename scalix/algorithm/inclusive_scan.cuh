
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
#include "../execute_kernel.cuh"
#include "functional.cuh"
#include <mutex>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace sclx::algorithm {

template<class T, class ResultT, uint Rank, class F>
__host__ void inclusive_scan(
    const array<const T, Rank>& arr,
    array<ResultT, Rank>& result,
    const T& identity,
    F&& f
) {
    if (arr.shape() != result.shape()) {
        throw std::invalid_argument(
            "Input and result arrays must have the same shape"
        );
    }

    result.unset_read_mostly();

    const auto& mem_info = arr.memory_info();
    std::vector<int> devices(
        mem_info.devices.get(),
        mem_info.devices.get() + mem_info.num_devices
    );

    std::vector<std::future<void>> futures;
    futures.reserve(devices.size());

    int current_device = cuda::traits::current_device();

    auto device_info = get_device_split_info(arr);

#ifndef __CLION_IDE__  // CLion shows incorrect errors with thrust
    // library
    for (auto& [_device_id, _slice_idx, _slice_size] : device_info) {

        auto lambda = [&](int device_id, size_t slice_idx, size_t slice_size) {
            cuda::set_device(device_id);  // init context for thread

            auto arr_slice
                = arr.get_range({slice_idx}, {slice_idx + slice_size});

            auto result_slice
                = result.get_range({slice_idx}, {slice_idx + slice_size});

            auto stream = cuda::stream_t::create_for_device(device_id);
            thrust::inclusive_scan(
                thrust::cuda::par.on(stream),
                arr_slice.begin(),
                arr_slice.end(),
                result_slice.begin(),
                f
            );
        };

        futures.emplace_back(std::async(
            std::launch::async,
            lambda,
            _device_id,
            _slice_idx,
            _slice_size
        ));
    }

    for (int d = 1; d < device_info.size(); d++) {
        auto& [device_id_prev, slice_idx_prev, slice_size_prev]
            = device_info[d - 1];
        auto& [device_id, slice_idx, slice_size] = device_info[d];

        auto arr_slice_prev = result.get_range(
            {slice_idx_prev},
            {slice_idx_prev + slice_size_prev}
        );

        futures[d - 1].wait();  // wait for previous device to finish
        auto& back_val = *(arr_slice_prev.end() - 1);

        auto arr_slice
            = result.get_range({slice_idx}, {slice_idx + slice_size});

        futures[d].wait();  // wait for current device to finish
        *(arr_slice.end() - 1) = f(back_val, *(arr_slice.end() - 1));
    }

    for (int _d = 1; _d < device_info.size(); _d++) {

        auto lambda = [&](int d) mutable {
            auto& [device_id_prev, slice_idx_prev, slice_size_prev]
                = device_info[d - 1];
            auto& [device_id, slice_idx, slice_size] = device_info[d];

            cuda::set_device(device_id);  // init context for thread

            auto arr_slice_prev = result.get_range(
                {slice_idx_prev},
                {slice_idx_prev + slice_size_prev}
            );

            auto arr_slice
                = result.get_range({slice_idx}, {slice_idx + slice_size});
            auto& back_val = *(arr_slice_prev.end() - 1);

            execute_kernel([&](kernel_handler& handler) {
                handler.launch(
                    md_range_t<Rank>(arr_slice.shape()),
                    arr_slice,
                    [=] __device__(auto& idx) {
                        if (&arr_slice[idx] == arr_slice.end() - 1)
                            return;
                        arr_slice[idx] = f(arr_slice[idx], back_val);
                    }
                );
            }).get();
        };

        futures.emplace_back(std::async(std::launch::async, lambda, _d));
    }

    for (auto& fut : futures) {
        fut.get();
    }

    cuda::set_device(current_device);

    result.set_read_mostly();
#endif
}

template<class T, class ResultT, uint Rank, class F>
__host__ void inclusive_scan(
    const array<T, Rank>& arr,
    array<ResultT, Rank>& result,
    const T& identity,
    F&& f
) {
    inclusive_scan(
        static_cast<const array<const T, Rank>&>(arr),
        result,
        identity,
        f
    );
}

template<class T, uint Rank, class F>
__host__ array<T, Rank>
inclusive_scan(const array<const T, Rank>& arr, const T& identity, F&& f) {
    array<T, Rank> result(arr.shape());

    inclusive_scan(arr, result, identity, f);

    return result;
}

template<class T, uint Rank, class F>
__host__ array<T, Rank>
inclusive_scan(const array<T, Rank>& arr, const T& identity, F&& f) {
    array<T, Rank> result(arr.shape());

    inclusive_scan(
        static_cast<const array<const T, Rank>&>(arr),
        result,
        identity,
        f
    );

    return result;
}

}  // namespace sclx::algorithm
