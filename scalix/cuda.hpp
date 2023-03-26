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
#include "shape.cuh"
#include "throw_exception.hpp"
#include <memory>
#include <stdexcept>

namespace sclx::cuda {

struct cuda_exception : std::runtime_error {
    explicit cuda_exception(const std::string& msg) : std::runtime_error(msg) {}

    static void raise_if_not_success(
        cudaError_t err,
        const std::experimental::source_location& location,
        const std::string& function_prefix = "sclx::cuda::"
    ) {
        if (err != cudaSuccess) {
            throw_exception<cuda_exception>(
                cudaGetErrorString(err),
                function_prefix,
                location
            );
        }
    }
};

void set_device(int device) {
#ifdef SCALIX_EMULATE_MULTIDEVICE
    device = 0;
#endif
    cudaError_t err = cudaSetDevice(device);
    cuda_exception::raise_if_not_success(
        err,
        std::experimental::source_location::current()
    );
}

struct traits {
    static constexpr int cpu_device_id = cudaCpuDeviceId;

    enum mem_advise {
        set_read_mostly          = cudaMemAdviseSetReadMostly,
        unset_read_mostly        = cudaMemAdviseUnsetReadMostly,
        set_preferred_location   = cudaMemAdviseSetPreferredLocation,
        unset_preferred_location = cudaMemAdviseUnsetPreferredLocation,
        set_accessed_by          = cudaMemAdviseSetAccessedBy,
        unset_accessed_by        = cudaMemAdviseUnsetAccessedBy
    };

    struct kernel {
      public:
        static constexpr shape_t<1> default_block_shape{256};
        static constexpr size_t default_grid_size
            = std::numeric_limits<size_t>::max();
    };

    static int device_count() {
#ifndef SCALIX_EMULATE_MULTIDEVICE
        int device_count;
        cudaGetDeviceCount(&device_count);
        return device_count;
#else
        return SCALIX_EMULATE_MULTIDEVICE;
#endif
    }

    static int current_device() {
        int device;
        cudaGetDevice(&device);
        return device;
    }
};

void mem_advise(
    void* ptr,
    size_t size,
    cuda::traits::mem_advise advice,
    int device
) {
#ifdef SCALIX_EMULATE_MULTIDEVICE
    device = 0;
#endif
    cudaError_t err = cudaMemAdvise(
        ptr,
        size,
        static_cast<cudaMemoryAdvise>(advice),
        device
    );
    cuda_exception::raise_if_not_success(
        err,
        std::experimental::source_location::current()
    );
}

template<class T>
void mem_prefetch_async(T* begin, T* end, int device, cudaStream_t stream = 0) {
#ifdef SCALIX_EMULATE_MULTIDEVICE
    device = 0;
#endif
    cudaError_t err = cudaMemPrefetchAsync(
        begin,
        std::distance(begin, end) * sizeof(T),
        device,
        stream
    );
    cuda_exception::raise_if_not_success(
        err,
        std::experimental::source_location::current()
    );
}

void stream_synchronize(cudaStream_t stream = 0) {
    cudaError_t err = cudaStreamSynchronize(stream);
    cuda_exception::raise_if_not_success(
        err,
        std::experimental::source_location::current()
    );
}

void peak_last_error_and_throw_if_error(
    const std::experimental::source_location& location,
    const std::string& function_prefix = "sclx::cuda::"
) {
    cudaError_t err = cudaPeekAtLastError();
    cuda_exception::raise_if_not_success(err, location, function_prefix);
}

void get_last_error_and_throw_if_error(
    const std::experimental::source_location& location,
    const std::string& function_prefix = "sclx::cuda::"
) {
    cudaError_t err = cudaGetLastError();
    cuda_exception::raise_if_not_success(err, location, function_prefix);
}

class stream_t {
  public:
    stream_t() = default;

    stream_t(const stream_t&)            = default;
    stream_t& operator=(const stream_t&) = default;

    stream_t(stream_t&&)            = default;
    stream_t& operator=(stream_t&&) = default;

    explicit stream_t(const cudaStream_t& stream) { *stream_ = stream; }

    static stream_t create() {
        cudaStream_t stream;
        cudaError_t err
            = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cuda_exception::raise_if_not_success(
            err,
            std::experimental::source_location::current()
        );
        return stream_t{stream};
    }

    static stream_t create_for_device(int device) {
        int current_device = traits::current_device();
        set_device(device);
        stream_t stream = create();
        set_device(current_device);
        return stream;
    }

    operator cudaStream_t() const {  // NOLINT(google-explicit-constructor)
        return *stream_;
    }

    [[nodiscard]] const cudaStream_t& get() const { return *stream_; }

    void synchronize() const {
        cudaError_t err = cudaStreamSynchronize(*stream_);
        cuda_exception::raise_if_not_success(
            err,
            std::experimental::source_location::current()
        );
    }

  private:
    std::shared_ptr<cudaStream_t> stream_ = std::shared_ptr<cudaStream_t>(
        new cudaStream_t{nullptr},
        [](cudaStream_t* stream) {
            cudaStreamDestroy(*stream);
            delete stream;
        }
    );
};

}  // namespace sclx::cuda
