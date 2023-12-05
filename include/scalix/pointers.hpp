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

namespace detail {
template<class>
constexpr bool is_unbounded_array_v = false;
template<class T>
constexpr bool is_unbounded_array_v<T[]> = true;

template<class>
constexpr bool is_bounded_array_v = false;
template<class T, std::size_t N>
constexpr bool is_bounded_array_v<T[N]> = true;
}  // namespace detail

template<class T>
class default_delete {
  public:
    default_delete()                       = default;
    default_delete(const default_delete&) = default;
    default_delete(default_delete&&)      = default;
    explicit default_delete(sycl::queue q) : q(std::move(q)) {}

    default_delete& operator=(const default_delete&) = default;
    default_delete& operator=(default_delete&&)      = default;

    void operator()(std::remove_extent_t<T>* ptr) const {
        sycl::free(ptr, q);
    }

  private:
    sycl::queue q;
};

template<class T, class Deleter = default_delete<T>>
using unique_ptr = std::unique_ptr<T, Deleter>;

template<class T>
using shared_ptr = std::shared_ptr<T>;

template<class T>
using weak_ptr = std::weak_ptr<T>;

template<class T, class... Args>
std::enable_if_t<!std::is_array<T>::value, sclx::unique_ptr<T>>
make_unique(sycl::queue q, ::sclx::usm::alloc alloc, Args&&... args) {
    auto ptr = unique_ptr<T>(
        sycl::malloc<T>(1, q, alloc),
        ::sclx::default_delete<T>{q}
    );
    auto host_val = std::make_unique<T>(std::forward<Args>(args)...);
    q.memcpy(ptr.get(), host_val.get(), sizeof(T)).wait_and_throw();
    return ptr;
}

template<class T>
std::enable_if_t<detail::is_unbounded_array_v<T>, sclx::unique_ptr<T>>
make_unique(sycl::queue q, ::sclx::usm::alloc alloc, std::size_t size) {
    return unique_ptr<T>(
        sycl::malloc<std::remove_extent_t<T>>(size, q, alloc),
        ::sclx::default_delete<T>{q}
    );
}

template<class T, class... Args>
std::enable_if_t<detail::is_bounded_array_v<T>>
make_unique(sycl::queue q, ::sclx::usm::alloc alloc, Args&&...) = delete;

template<class T, class... Args>
shared_ptr<T>
make_shared(sycl::queue q, ::sclx::usm::alloc alloc, Args&&... args) {
    return shared_ptr<T>(
        make_unique<T>(q, alloc, std::forward<Args>(args)...).release(),
        ::sclx::default_delete<T>{q}
    );
}

}  // namespace sclx
