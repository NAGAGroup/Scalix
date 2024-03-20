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

namespace sclx {

template<class R>
auto typed_task<R>::get_future() -> std::future<R> {
    return std::move(future_);
}

template<class R>
typed_task<R>::typed_task(std::unique_ptr<impl>&& impl, std::future<R>&& future)
    : generic_task{std::move(impl)},
      future_{std::move(future)} {}

template<class R>
template<class... Args>
class typed_task<R>::typed_impl<R(Args...)> final : public impl {
  public:
    template<class F>
    explicit typed_impl(F&& func)  // cppcheck-suppress // NOLINT
        : task_{std::make_shared<std::packaged_task<R()>>(std::forward<F>(func)
        )},
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
    typed_impl(F&& func, PassedArg1&& arg1, PassedArgs&&... args)
        : task_{std::make_shared<std::packaged_task<R(Args...)>>(
            std::forward<F>(func)
        )},
          args_ptr_{
              std::make_shared<std::tuple<std::remove_reference_t<Args>...>>(
                  std::forward<PassedArg1>(arg1),
                  std::forward<PassedArgs>(args)...
              )} {
        static_assert(
            sizeof...(PassedArgs) + 1 == sizeof...(Args),
            "provided task arguments in constructor do not match task signature"
        );
    }

    [[nodiscard]] auto get_future() const -> std::future<R> {
        return task_->get_future();
    }

    void async_execute(std::shared_ptr<task_metadata> metadata) const override {
        auto& args_ptr = args_ptr_;
        auto& task     = task_;
        std::thread exec_thread{
            [metadata = std::move(metadata), args_ptr, task] {
                typed_impl::apply(*task, *args_ptr);
                const std::lock_guard lock(metadata->mutex);
                for (const auto& dependent_task : metadata->dependent_tasks) {
                    dependent_task.impl_->decrease_dependency_count(
                        dependent_task.metadata_
                    );
                }
                metadata->has_completed = true;
            }};
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

struct task_factory {
    template<class F, class... Args>
    static auto create_task(F&& func, Args&&... args)
        -> typed_task<std::invoke_result_t<F, Args...>> {
        using task_t          = typed_task<std::invoke_result_t<F, Args&&...>>;
        using specification_t = typename task_t::template typed_impl<
            std::invoke_result_t<F, Args&&...>(Args && ...)>;
        auto impl_ptr = std::make_unique<specification_t>(
            std::forward<F>(func),
            std::forward<Args>(args)...
        );
        auto future = impl_ptr->get_future();
        return task_t{std::move(impl_ptr), std::move(future)};
    }
};

template<class F, class... Args>
auto create_task(F&& func, Args&&... args)
    -> typed_task<std::invoke_result_t<F, Args...>> {
    return task_factory::create_task<F, Args...>(
        std::forward<F>(func),
        std::forward<Args>(args)...
    );
}
}  // namespace sclx
