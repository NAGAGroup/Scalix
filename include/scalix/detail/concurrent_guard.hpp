// BSD 3-Clause License
//
// Copyright (c) 2024 Jack Myers
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
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace sclx::detail {

template<class T>
class concurrent_guard;

template<class T>
class exclusive_view {
    using lock_type = std::conditional_t<
        std::is_const_v<T>,
        std::shared_lock<std::shared_mutex>,
        std::unique_lock<std::shared_mutex>>;

  public:
    friend class concurrent_guard<T>;
    auto access() const -> T& {
        if (!lock_.owns_lock()) {
            throw std::runtime_error(
                "exclusive_view has been released"
            );
        }
        return *ptr_;
    }

    void release() {
        lock_.unlock();
    }

    void lock() {
        lock_.lock();
    }

  private:
    exclusive_view(
        std::shared_ptr<T> ptr,
        std::shared_ptr<std::shared_mutex> mutex,
        lock_type lock
    )
        : ptr_(std::move(ptr)),
          mutex_(std::move(mutex)),
          lock_(std::move(lock)) {
        lock_.lock();
    }

    std::shared_ptr<T> ptr_;
    std::shared_ptr<std::shared_mutex> mutex_;
    lock_type lock_;
};

template<class T>
class concurrent_guard {
  public:
    concurrent_guard() = default;
    explicit concurrent_guard(std::unique_ptr<T> ptr) : ptr_(std::move(ptr)) {}

    concurrent_guard(const concurrent_guard&)                    = default;
    auto operator=(const concurrent_guard&) -> concurrent_guard& = default;

    concurrent_guard(concurrent_guard&&)                    = default;
    auto operator=(concurrent_guard&&) -> concurrent_guard& = default;

    auto get_view() const -> exclusive_view<T> {
        return get_view_generic<T>();
    }

    auto get_const_view() const -> exclusive_view<const T> {
        return get_view_generic<const T>();
    }

    ~concurrent_guard() = default;

  private:

    template <class U>
    auto get_view_generic () const -> exclusive_view<U>
    {
        if (!ptr_) {
            throw std::runtime_error(
                "concurrent_guard does not hold valid data"
            );
        }

        using lock_type = typename exclusive_view<U>::lock_type;

        lock_type lock(*mutex_, std::defer_lock);

        return exclusive_view<U>{
            ptr_,
            mutex_,
            std::move(lock)
        };
    }
    std::shared_ptr<T> ptr_;
    std::shared_ptr<std::shared_mutex> mutex_ = std::make_shared<std::shared_mutex>();
};

}  // namespace sclx::detail
