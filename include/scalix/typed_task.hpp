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
#include "detail/generic_task.hpp"
#include "generic_task.hpp"
#include "typed_task_interface.hpp"
#include <cstdint>
#include <future>
#include <memory>
#include <scalix/concurrent_guard.hpp>
#include <scalix/defines.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sclx {

template<class R>
template<class... Args>
class typed_task<R>::typed_impl final : public impl {
  public:
    using function_type
        = std::conditional_t<sizeof...(Args) == 0, R(), R(Args...)>;

    template<class F>
    explicit typed_impl(F&& func)  // cppcheck-suppress // NOLINT
        : task_{std::make_shared<std::packaged_task<function_type>>(
              std::forward<F>(func)
          )},
          args_ptr_{std::make_shared<std::tuple<>>()} {
        future_ = std::move(task_->get_future());
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
        : args_ptr_{std::make_shared<std::tuple<
              std::remove_reference_t<PassedArg1>,
              std::remove_reference_t<PassedArgs>...>>(
              std::forward<PassedArg1>(arg1),
              std::forward<PassedArgs>(args)...
          )},
          task_{std::make_shared<std::packaged_task<function_type>>(
              std::forward<F>(func)
          )} {
        future_ = std::move(task_->get_future());
        static_assert(
            sizeof...(PassedArgs) + 1 == sizeof...(Args),
            "provided task arguments in constructor do not match task signature"
        );
    }

    void async_execute() const override {
        auto& args_ptr      = args_ptr_;
        auto& task          = task_;
        auto metadata_guard = metadata_;

        typed_impl::apply(*task, *args_ptr);
        {
            auto metadata
                = metadata_guard.template get_view<access_mode::write>();
            metadata.access().has_completed = true;
        }
        {
            auto metadata
                = metadata_guard.template get_view<access_mode::read>();
            for (const auto& dependent_task :
                 metadata.access().dependent_tasks) {
                dependent_task.impl_->decrease_dependency_count();
            }
        }
        //        std::thread exec_thread{
        //            [metadata_guard, args_ptr, task] {
        //                typed_impl::apply(*task, *args_ptr);
        //                {
        //                    auto metadata = metadata_guard.template
        //                    get_view<access_mode::write>();
        //                    metadata.access().has_completed = true;
        //                }
        //                {
        //                    auto metadata = metadata_guard.template
        //                    get_view<access_mode::read>(); for (const auto&
        //                    dependent_task :
        //                         metadata.access().dependent_tasks) {
        //                        dependent_task.impl_->decrease_dependency_count();
        //                    }
        //                }
        //            }
        //        };
        //        exec_thread.detach();
    }

    template<std::uint8_t N = 0, class Task, class Tuple, class... OArgs>
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

    std::shared_ptr<std::packaged_task<function_type>> task_;
    std::shared_ptr<std::tuple<std::remove_reference_t<Args>...>> args_ptr_;
    static constexpr auto is_rvalue_ref_args_
        = std::make_tuple(std::is_rvalue_reference_v<Args>...);
    std::future<R> future_{};
};

struct task_factory {
    template<class F, class... Args>
    static auto create_task(F&& func, Args&&... args)
        -> typed_task<std::invoke_result_t<F, Args...>> {
        using task_t       = typed_task<std::invoke_result_t<F, Args...>>;
        using typed_impl_t = typename task_t::template typed_impl<
            decltype(std::forward<Args>(args))...>;
        return task_t{std::make_unique<typed_impl_t>(
            std::forward<F>(func),
            std::forward<Args>(args)...
        )};
    }
};

template<class R>
auto typed_task<R>::get_future() -> std::future<R> {
    return std::move(*future_ptr_);
}

template<class R>
template<class... Args>
typed_task<R>::typed_task(std::unique_ptr<typed_impl<Args...>> impl)
    : generic_task{std::move(impl)},
      future_ptr_(&static_cast<decltype(impl.get())>(this->impl_.get())->future_
      ) {}


template<class R>
typed_task<R>::operator generic_task() {
    return {impl_};
}

template<class F, class... Args>
auto create_task(F&& func, Args&&... args)
    -> typed_task<std::invoke_result_t<F, Args...>> {
    return task_factory::create_task(
        std::forward<F>(func),
        std::forward<Args>(args)...
    );
}
}  // namespace sclx
