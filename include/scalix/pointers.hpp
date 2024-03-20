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

#include "defines.hpp"
#include <memory>
#include <utility>

namespace sclx {

template<class T>
using array_type = T[];  // NOLINT(*-avoid-c-arrays)

template<class T, std::size_t N>
using bounded_array_type = T[N];  // NOLINT(*-avoid-c-arrays)

namespace detail {
template<class>
constexpr bool is_unbounded_array_v = false;
template<class T>
constexpr bool is_unbounded_array_v<array_type<T>> = true;

template<class>
constexpr bool is_bounded_array_v = false;
template<class T, std::size_t N>
constexpr bool is_bounded_array_v<bounded_array_type<T, N>> = true;
}  // namespace detail

template<class T>
class default_delete {
  public:
    default_delete()                          = default;
    default_delete(const default_delete&)     = default;
    default_delete(default_delete&&) noexcept = default;
    explicit default_delete(const sycl::queue& queue) : queue_(queue) {}

    auto operator=(const default_delete&) -> default_delete&     = default;
    auto operator=(default_delete&&) noexcept -> default_delete& = default;

    void operator()(std::remove_extent_t<T>* ptr) const {
        sycl::free(ptr, queue_);
    }

    ~default_delete() = default;

  private:
    sycl::queue queue_;
};

template<class T, class Deleter = default_delete<T>>
using unique_ptr = std::unique_ptr<T, Deleter>;

template<class T>
using shared_ptr = std::shared_ptr<T>;

template<class T>
using weak_ptr = std::weak_ptr<T>;

template<class T, class... Args>
auto make_unique(sycl::queue queue, ::sclx::usm::alloc alloc, Args&&... args)
    -> std::enable_if_t<!std::is_array_v<T>, sclx::unique_ptr<T>> {
    auto ptr = unique_ptr<T>(
        sycl::malloc<T>(1, queue, alloc),
        ::sclx::default_delete<T>{queue}
    );
    auto host_val = std::make_unique<T>(std::forward<Args>(args)...);
    queue.memcpy(ptr.get(), host_val.get(), sizeof(T)).wait_and_throw();
    return ptr;
}

template<class T>
auto make_unique(sycl::queue queue, ::sclx::usm::alloc alloc, std::size_t size)
    -> std::enable_if_t<detail::is_unbounded_array_v<T>, sclx::unique_ptr<T>> {
    auto ptr = unique_ptr<T>(
        sycl::malloc<std::remove_extent_t<T>>(size, queue, alloc),
        ::sclx::default_delete<T>{queue}
    );
    auto host_val = std::make_unique<T>(size);
    queue
        .memcpy(
            ptr.get(),
            host_val.get(),
            sizeof(std::remove_extent_t<T>) * size
        )
        .wait_and_throw();
    return ptr;
}

template<class T>
auto make_unique(sycl::queue queue, ::sclx::usm::alloc alloc)
    -> std::enable_if_t<detail::is_bounded_array_v<T>> {
    auto ptr = unique_ptr<T>(
        sycl::malloc<T>(1, queue, alloc),
        ::sclx::default_delete<T>{queue}
    );
    auto host_val = std::make_unique<T>();
    queue.memcpy(ptr.get(), host_val.get(), sizeof(T)).wait_and_throw();
    return ptr;
}

template<class T, class... Args>
auto make_shared(sycl::queue queue, ::sclx::usm::alloc alloc, Args&&... args)
    -> shared_ptr<T> {
    return shared_ptr<T>(
        make_unique<T>(queue, alloc, std::forward<Args>(args)...).release(),
        ::sclx::default_delete<T>{queue}
    );
}

}  // namespace sclx
