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

#include <future>
#include <scalix/generic_task.hpp>

namespace sclx {
template<class R>
class typed_task : public generic_task {
    friend class generic_task;
public:
    [[nodiscard]] auto get_future() const -> std::future<R> {
        return future_accessor_->get_future(impl_.get());
    }

private:
    using generic_task::generic_task;

    struct generic_future_accessor {
        generic_future_accessor() = default;

        generic_future_accessor(const generic_future_accessor&) = delete;
        auto operator=(const generic_future_accessor&)
            -> generic_future_accessor&                    = delete;
        generic_future_accessor(generic_future_accessor&&) = delete;
        auto operator=(generic_future_accessor&&)
            -> generic_future_accessor& = delete;

        virtual ~generic_future_accessor() = default;
        virtual auto get_future(void* spec_ptr) const -> std::future<R> = 0;
    };

    template<class... Args>
    struct future_accessor final : generic_future_accessor {
        auto get_future(void* spec_ptr) const -> std::future<R> override {
            auto spec = static_cast<specification<R(Args...)>*>(spec_ptr);
            return spec->get_future();
        }
    };

    std::unique_ptr<generic_future_accessor> future_accessor_
        = std::make_unique<future_accessor<>>();
};

template<class R, class... Args>
class generic_task::specification<R(Args...)> final : public impl {
  public:
    template<class F>
    specification(F&& f)
        : task_{std::make_shared<std::packaged_task<R()>>(std::forward<F>(f))},
          args_ptr_{std::make_shared<std::tuple<>>()} {
        static_assert(
            !std::is_base_of_v<impl, F>,
            "task cannot be constructed from another task"
        );
        static_assert(
            sizeof...(Args) == 0,
            "provided task arguments in constructor do not match task signature"
        );
    }

    template<class F, class PassedArg1, class... PassedArgs>
    specification(F&& f, PassedArg1&& arg1, PassedArgs&&... args)
        : task_{std::make_shared<std::packaged_task<R(Args...)>>(
              std::forward<F>(f)
          )},
          args_ptr_{
              std::make_shared<std::tuple<std::remove_reference_t<Args>...>>(
                  std::forward<PassedArg1>(arg1),
                  std::forward<PassedArgs>(args)...
              )
          } {
        static_assert(
            sizeof...(PassedArgs) + 1 == sizeof...(Args),
            "provided task arguments in constructor do not match task signature"
        );
    }

    [[nodiscard]] auto get_future() const -> std::future<R> {
        return task_->get_future();
    }

  private:
    void async_execute(std::shared_ptr<task_metadata> metadata) const override {
        auto& args_ptr = args_ptr_;
        auto& task     = task_;
        std::thread exec_thread{
            [metadata = std::move(metadata), args_ptr, task] {
                specification::apply(*task, *args_ptr);
                const std::lock_guard lock(metadata->mutex);
                for (const auto& dependent_task : metadata->dependent_tasks) {
                    dependent_task.impl_->decrease_dependency_count(
                        dependent_task.metadata_
                    );
                }
                metadata->has_completed = true;
            }
        };
        exec_thread.detach();
    }

    template<uint N = 0, class Task, class Tuple, class... OArgs>
    static void apply(Task& task, Tuple& args, OArgs&&... oargs) {
        if constexpr (sizeof...(Args) == 0) {
            return task();
        }

        if constexpr (N == sizeof...(Args)) {
            return task(std::forward<OArgs>(oargs)...);
        } else {
            if constexpr (std::get<N>(is_rvalue_ref_args_)) {
                return apply<N + 1>(
                    task,
                    args,
                    std::forward<OArgs>(oargs)...,
                    std::move(std::get<N>(args))
                );
            } else {
                return apply<N + 1>(
                    task,
                    args,
                    std::forward<OArgs>(oargs)...,
                    std::get<N>(args)
                );
            }
        }
    }

    std::shared_ptr<std::packaged_task<R(Args...)>> task_;
    std::shared_ptr<std::tuple<std::remove_reference_t<Args>...>> args_ptr_;
    static constexpr auto is_rvalue_ref_args_
        = std::make_tuple(std::is_rvalue_reference_v<Args>...);
};

template<class F, class... Args>
auto generic_task::create(F&& func, Args&&... args)
    -> typed_task<std::invoke_result_t<F, Args...>> {
    return typed_task<std::invoke_result_t<F, Args&&...>>{std::make_unique<
        specification<std::invoke_result_t<F, Args&&...>(Args&&...)>>(
        std::forward<F>(func),
        std::forward<Args>(args)...
    )};
}

}
