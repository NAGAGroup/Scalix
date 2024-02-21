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
#include <scalix/detail/concurrent_guard.hpp>
#include <scalix/detail/state_machine.hpp>

namespace sclx::detail {

template<class Func>
concept StateBasedActionType = requires(Func&& func) {
    { func() } -> std::same_as<void>;
};

template<class StateDefinition>
class state_based_scheduler {
  public:
    using state_machine_type = state_machine<StateDefinition>;
    using state_label =
        typename state_machine_traits<state_machine_type>::state_label;
    using inputs_type =
        typename state_machine_traits<state_machine_type>::inputs_type;

  private:
    struct action_wrapper_base {
        action_wrapper_base()                           = default;
        action_wrapper_base(const action_wrapper_base&) = delete;
        action_wrapper_base(action_wrapper_base&&)      = delete;
        auto operator=(const action_wrapper_base&)
            -> action_wrapper_base&                                   = delete;
        auto operator=(action_wrapper_base&&) -> action_wrapper_base& = delete;

        virtual void operator()() const = 0;
        virtual ~action_wrapper_base()  = default;
    };

    struct scheduler_metadata {
        state_machine_type state_machine_;
        std::queue<std::unique_ptr<action_wrapper_base>> action_queue_;
    };

    template<class Action, class... Args>
    struct action_wrapper final : action_wrapper_base {
        Action action;
        std::tuple<Args...> args;

        explicit action_wrapper(
            Action&& action,  // NOLINT(*-rvalue-reference-param-not-moved)
            Args&&... args
        )
            : action{std::forward<Action>(action)},
              args{std::forward<Args>(args)...} {}

        void operator()() const override { std::apply(action, args); }
    };

    struct action_definition {
        state_label entry_requirement;
        inputs_type inputs_on_test_entry;
        inputs_type inputs_on_exit;
        std::unique_ptr<action_wrapper_base> action;
    };

  public:
    template<class Action, class... Args>
    auto submit(
        const state_label& entry_requirement,
        inputs_type inputs_on_test_entry,
        inputs_type inputs_on_exit,
        Action&& action,
        Args&&... args
    ) -> sclx::event {
        using action_wrapper_type = action_wrapper<Action, Args...>;

        auto metadata_view = metadata_guard_.get_view();
        auto queue_empty_on_submit
            = metadata_view.access().action_queue_.empty();

        auto original_action = std::make_unique<action_wrapper_type>(
            std::forward<Action>(action),
            std::forward<Args>(args)...
        );

        std::promise<void> action_promise;
        auto action_future   = action_promise.get_future();
        auto promised_action = std::make_unique<action_wrapper_type>(
            [original_action = std::move(original_action),
             action_promise  = std::move(action_promise)] mutable {
                (*original_action)();
                action_promise.set_value();
            }
        );
        auto promised_action_def = action_definition{
            entry_requirement,
            inputs_on_test_entry,
            inputs_on_exit,
            std::move(promised_action)
        };
        metadata_view.access().action_queue_.emplace(
            std::move(promised_action_def)
        );
        return queue_.submit([&](sycl::handler& cgh) {
            host_task(cgh, [action_future = std::move(action_future)] {
                action_future.wait();
            });
        });
    }

    auto get_next_action() const -> action_definition& {
        return metadata_guard_.get_view().access().action_queue_.front();
    }

    auto pop_action() -> void {
        metadata_guard_.get_view().access().action_queue_.pop();
    }

    auto execute_next_action() -> sclx::event {
        auto event = queue_.submit([&](sycl::handler& cgh) {
            const auto& metadata_guard       = metadata_guard_;
            auto& action_def                 = get_next_action();
            bool requirement_satisfied       = false;
            const auto& entry_requirement    = action_def.entry_requirement;
            const auto& inputs_on_test_entry = action_def.inputs_on_test_entry;
            while (!requirement_satisfied) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                auto metadata_view = metadata_guard.get_view();
                metadata_view.access().state_machine_.transition_states(
                    inputs_on_test_entry
                );
                requirement_satisfied
                    = metadata_view.access().state_machine_.current_state()
                   == entry_requirement;
            }
            host_task(
                cgh,
                [metadata_guard, action_def = std::move(action_def)] {
                    const auto& action_ptr = action_def.action;
                    (*action_ptr)();
                    const auto& inputs_on_exit = action_def.inputs_on_exit;
                    auto metadata_view         = metadata_guard.get_view();
                    metadata_view.access().state_machine_.transition_states(
                        inputs_on_exit
                    );
                }
            );
        });

        pop_action();
        if (!metadata_guard_.get_view().access().action_queue_.empty()) {
            execute_next_action();
        }

        return event;
    }

  private:
    concurrent_guard<scheduler_metadata> metadata_guard_{
        std::make_unique<scheduler_metadata>()
    };

    sycl::queue queue_;
};

};  // namespace sclx::detail
