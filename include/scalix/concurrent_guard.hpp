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
#include <scalix/defines.hpp>
#include <shared_mutex>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>

namespace sclx {

namespace detail {

struct access_queue {
    using tag_t     = std::unique_ptr<unsigned char>;
    using raw_tag_t = unsigned char*;
    std::deque<raw_tag_t> view_tags;
    std::mutex mtx;
    static auto create_tag() -> tag_t {
        return std::make_unique<unsigned char>();
    }
};

}  // namespace detail

template<class T>
    requires std::is_same_v<T, std::decay_t<T>>
class concurrent_guard;

template<class T>
class concurrent_view {
    friend class concurrent_guard<std::remove_const_t<T>>;
    using lock_type = std::conditional_t<
        std::is_const_v<T>,
        std::shared_lock<std::shared_mutex>,
        std::unique_lock<std::shared_mutex>>;

  public:
    using value_type = T;

    concurrent_view(const concurrent_view&)                    = delete;
    auto operator=(const concurrent_view&) -> concurrent_view& = delete;

    concurrent_view(concurrent_view&&)                    = default;
    auto operator=(concurrent_view&&) -> concurrent_view& = default;

    [[nodiscard]] auto access() const& -> T& {
        if (!lock_.owns_lock()) {
            throw std::runtime_error("concurrent_view has been unlocked");
        }
        return *ptr_;
    }

    auto operator->() const -> T* { return &access(); }

    auto access() && -> T& = delete;

    void unlock() { lock_.unlock(); }

    void lock() {
        enqueue_access();
        bool can_access = false;
        while (!can_access) {
            detail::access_queue::raw_tag_t tag_front = nullptr;
            {
                std::lock_guard lock{access_queue_->mtx};
                tag_front = access_queue_->view_tags.front();
            }
            can_access = (tag_front == tag_.get());
        }
        std::lock_guard lock{access_queue_->mtx};
        access_queue_->view_tags.pop_front();
        lock_.lock();
    }

    ~concurrent_view() = default;

  private:
    concurrent_view(
        std::shared_ptr<T> ptr,
        std::shared_ptr<std::shared_mutex> mutex,
        std::shared_ptr<detail::access_queue> access_queue,
        std::defer_lock_t defer_lock
    )
        : ptr_(std::move(ptr)),
          mutex_(mutex),
          lock_(lock_type(*mutex, defer_lock)),
          access_queue_(std::move(access_queue)) {
        enqueue_access();
    }

    concurrent_view(
        std::shared_ptr<T> ptr,
        std::shared_ptr<std::shared_mutex> mutex,
        std::shared_ptr<detail::access_queue> access_queue
    )
        : concurrent_view(
              std::move(ptr),
              std::move(mutex),
              std::move(access_queue),
              std::defer_lock
          ) {
        lock();
    }

    void enqueue_access() {
        {
            std::lock_guard lock{access_queue_->mtx};
            if (std::find(
                    access_queue_->view_tags.begin(),
                    access_queue_->view_tags.end(),
                    tag_.get()
                )
                == access_queue_->view_tags.end()) {
                access_queue_->view_tags.push_back(tag_.get());
            }
        }
    }
    std::shared_ptr<T> ptr_;
    std::shared_ptr<std::shared_mutex> mutex_;
    lock_type lock_;
    std::shared_ptr<detail::access_queue> access_queue_;
    detail::access_queue::tag_t tag_ = detail::access_queue::create_tag();
};

template<class T, access_mode Mode>
struct concurrent_view_type_get {
    using type = std::conditional_t<Mode == access_mode::read, const T, T>;

    static_assert(
        Mode != access_mode::discard_write
            && Mode != access_mode::discard_read_write
            && Mode != access_mode::atomic,
        "Invalid access mode for concurrent_view by using deprecated access "
        "mode"
    );
};

template<class T, access_mode Mode>
using concurrent_view_t
    = concurrent_view<typename concurrent_view_type_get<T, Mode>::type>;

template<class T>
    requires std::is_same_v<T, std::decay_t<T>>
class concurrent_guard {
  public:
    template<class U>
        requires std::is_same_v<U, std::decay_t<U>>
    friend class concurrent_guard;

    template<class U>
    concurrent_guard(const concurrent_guard<U>& other)
        : ptr_(other.ptr_),
          mutex_(other.mutex_) {}

    concurrent_guard() : ptr_(std::make_shared<T>()) {}

    explicit concurrent_guard(T value)
        : ptr_(std::make_shared<T>(std::move(value))) {}

    explicit concurrent_guard(std::shared_ptr<T> ptr) : ptr_(std::move(ptr)) {}

    concurrent_guard(const concurrent_guard&)                    = default;
    auto operator=(const concurrent_guard&) -> concurrent_guard& = default;

    concurrent_guard(concurrent_guard&&)                    = default;
    auto operator=(concurrent_guard&&) -> concurrent_guard& = default;

    template<access_mode Mode = access_mode::read>
    [[nodiscard]] auto get_view() const -> concurrent_view_t<T, Mode> {
        return get_view_generic<
            typename concurrent_view_t<T, Mode>::value_type>();
    }

    template<access_mode Mode = access_mode::read>
    [[nodiscard]] auto get_view(std::defer_lock_t defer_lock
    ) const -> concurrent_view_t<T, Mode> {
        return get_view_generic<
            typename concurrent_view_t<T, Mode>::value_type>(defer_lock);
    }

    [[nodiscard]] auto valid() const -> bool { return ptr_ != nullptr; }

    [[nodiscard]] auto unsafe_access() const -> T& {
        if (!valid()) {
            throw std::runtime_error("concurrent_guard does not hold valid data"
            );
        }
        return *ptr_;
    }

    ~concurrent_guard() = default;

  private:
    template<class U>
    [[nodiscard]] [[nodiscard]] auto
    get_view_generic() const -> concurrent_view<U> {
        if (!valid()) {
            throw std::runtime_error("concurrent_guard does not hold valid data"
            );
        }

        return {ptr_, mutex_, access_queue_};
    }
    template<class U>
    [[nodiscard]] [[nodiscard]] auto
    get_view_generic(std::defer_lock_t defer_lock) const -> concurrent_view<U> {
        if (!valid()) {
            throw std::runtime_error("concurrent_guard does not hold valid data"
            );
        }

        return {ptr_, mutex_, access_queue_, defer_lock};
    }
    std::shared_ptr<T> ptr_ = std::make_shared<T>();
    std::shared_ptr<std::shared_mutex> mutex_
        = std::make_shared<std::shared_mutex>();
    std::shared_ptr<detail::access_queue> access_queue_
        = std::make_shared<detail::access_queue>();
};

}  // namespace sclx
