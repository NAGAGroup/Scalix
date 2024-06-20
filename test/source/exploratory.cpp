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
#include <scalix/access_anchor.hpp>
int main() {
    sclx::queue q;
    std::vector<sycl::queue> device_queues;
    auto device_list = sycl::device::get_devices();
    for (auto& device : device_list) {
        if (device.is_cpu()) {
            device_queues.emplace_back(device);
        }
    }
    q.device_queues_ = std::move(device_queues);
    auto num_devices = q.device_queues_.size();
    std::vector<double> device_weights;
    std::transform(
        q.device_queues_.begin(),
        q.device_queues_.end(), std::back_inserter(device_weights), [&](const sycl::queue& device_queue) {
            return 1.0 / static_cast<double>(num_devices);
        });
    q.device_weights_ = std::move(device_weights);
    sclx::buffer<float, 1> buffer{10 * num_devices};
    q.submit([&](sclx::handler& cgh) {
        auto buffer_acc = buffer.get_access<sycl::access_mode::write>(cgh, sclx::default_access_strategy{});
        cgh.parallel_for(sycl::range<1>{10 * num_devices}, [=](sycl::id<1> idx) {
            buffer_acc[idx] = static_cast<float>(idx[0]);
        });
    });
}