// BSD 3-Clause License
//
// Copyright (c) 2023-2024 Jack Myers
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
#include <scalix/buffer.hpp>
#include <scalix/defines.hpp>
int main() {
    std::vector<sycl::queue> device_queues;
    auto device_list = sycl::device::get_devices();
    for (auto& device : device_list) {
        if (device.is_gpu()) {
            device_queues.emplace_back(device);
            break;
        }
    }
    auto num_devices = device_queues.size();
    std::vector<uint> device_weights{1};
    sclx::queue dist_queue{device_weights, device_queues};
    sclx::buffer<double, 1> buffer{10 * num_devices};
    sycl::buffer<double, 1> buffer_host{10 * num_devices};

    dist_queue.submit([&](sclx::command_handler& cgh) {
        auto acc = buffer.get_access<sclx::access_mode::write>(cgh);
        cgh.parallel_for(sclx::range<>{10 * num_devices}, [=](sycl::id<> idx) {
            // acc[idx] = idx[0];
        });
    });

    // dist_queue.submit([&](sclx::command_handler& cgh) {
    //     auto acc
    //         =
    //         buffer_host.get_access<sycl::access_mode::write>(*cgh.sycl_handler
    //         );
    //     auto acc_src = buffer.get_access<sclx::access_mode::read>(cgh);
    //     cgh.parallel_for(sclx::range<>{10 * num_devices}, [=](sycl::id<> idx)
    //     {
    //         // acc[idx] = acc_src[idx];
    //     });
    // });
    //
    // auto host_acc = buffer_host.get_access<sycl::access_mode::read>();
    // for (auto& val : host_acc) {
    //     std::cout << val << " ";
    // }

    return 0;
}
