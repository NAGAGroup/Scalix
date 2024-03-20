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
    struct action_metadata {
        using valid_action_t = bool;
        std::promise<void> action_signal;
        valid_action_t action_signal_valid;

        action_metadata(
            std::promise<void> action_signal,
            const valid_action_t& action_signal_valid
        )
            : action_signal{std::move(action_signal)},
              action_signal_valid{action_signal_valid} {}
    };

    struct scheduler_metadata {
        state_machine_type state_machine_;
        using valid_action_t = bool;
        std::queue<action_metadata> action_signal_queue_;
    };

    template<class Action, class... Args>
    struct action_wrapper {
        Action action;
        std::tuple<Args...> args;

        explicit action_wrapper(Action&& action, Args&&... args)
            : action{std::move<Action>(action)},
              args{std::forward<Args>(args)...} {}

        action_wrapper(const action_wrapper&)                    = delete;
        auto operator=(const action_wrapper&) -> action_wrapper& = delete;
        action_wrapper(action_wrapper&&)                         = default;
        auto operator=(action_wrapper&&) -> action_wrapper&      = default;

        void operator()() const { std::apply(action, args); }

        ~action_wrapper() = default;
    };

    template<class Action, class... Args>
    struct action_definition {
        state_label entry_requirement;
        inputs_type inputs_on_test_entry;
        inputs_type inputs_on_exit;
        action_wrapper<Action, Args...> action;
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

        auto wrapped_action = action_wrapper_type{
            std::forward<Action>(action),
            std::forward<Args>(args)...};

        std::promise<void> action_signal;
        auto action_signal_future = action_signal.get_future();
        auto promised_action_def  = action_definition{
            entry_requirement,
            inputs_on_test_entry,
            inputs_on_exit,
            std::move(wrapped_action)};
        metadata_view.access().action_signal_queue_.emplace(
            std::move(action_signal),
            true
        );

        auto action_event = queue_.submit([&](sycl::handler& cgh) {
            const auto& metadata_guard = metadata_guard_;
            host_task(
                cgh,
                [action_signal_future = std::move(action_signal_future),
                 promised_action_def  = std::move(promised_action_def),
                 metadata_guard] mutable {
                    // wait for the action to be signaled
                    // by the scheduler
                    action_signal_future.get();

                    auto action_metadata_view = metadata_guard.get_view();
                    action_metadata_view.unlock();
                    while (true) {
                        action_metadata_view.lock();

                        auto& metadata = action_metadata_view.access();

                        // the state machine is transitioned each loop
                        // by the inputs testing the entry requirement
                        //
                        // making the entry requirement a callable was
                        // considered, in which case the state machine would've
                        // only been transitioned if the requirement was
                        // satisfied which would occur once another running
                        // action transitioned the state machine to a valid
                        // state on exit. this would allow actions to have entry
                        // requirements that took a union of different states
                        //
                        // but it made more sense to keep the entry requirement
                        // as a simple state label check, instead offloading
                        // the logic to the state machine itself, keeping
                        // the spirit of a state-based scheduler intact
                        metadata.state_machine_.transition_states(
                            promised_action_def.inputs_on_test_entry
                        );

                        if (metadata.state_machine_.current_state()
                            == promised_action_def.entry_requirement) {
                            break;
                        }

                        action_metadata_view.unlock();

                        std::this_thread::yield();
                    }

                    action_metadata_view.access().action_signal_queue_.pop();
                    action_metadata_view.unlock();
                    promised_action_def.action();

                    action_metadata_view.lock();
                    action_metadata_view.access()
                        .state_machine_.transition_states(
                            promised_action_def.inputs_on_exit
                        );
                }
            );
        });

        // by calling execute action for every action submitted
        // we can ensure that each action will get executed, even if
        // the following call doesn't execute the action submitted
        // by the current call
        execute_next_action();

        return action_event;
    }

  private:
    void execute_next_action() {
        auto metadata_view = metadata_guard_.get_view();
        metadata_view.unlock();
        while (true) {
            metadata_view.lock();
            auto& metadata = metadata_view.access();

            if (auto& action_signal_queue = metadata.action_signal_queue_;
                action_signal_queue.front().action_signal_valid) {
                action_signal_queue.front().action_signal.set_value();
                action_signal_queue.front().action_signal_valid = false;
                return;
            }

            metadata_view.unlock();

            std::this_thread::yield();
        }
    }

    concurrent_guard<scheduler_metadata> metadata_guard_{
        std::make_unique<scheduler_metadata>()};

    sycl::queue queue_;
};

}  // namespace sclx::detail
