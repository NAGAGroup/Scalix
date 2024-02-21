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
#include <scalix/defines.hpp>

namespace sclx::detail {

template<class StateInputs, class StateLabel>
class state_definition;

template<class StateInputs, class StateLabel>
class state_transition {
  public:
    using state_type  = state_definition<StateInputs, StateLabel>;
    using result_type = std::unique_ptr<state_type>;
    virtual auto
    execute(const StateInputs& inputs, const state_type& current_state)
        -> result_type
        = 0;
    virtual ~state_transition() = default;
};

struct state_machine_invalid_label_type {};

template<class StateOrStateMachine>
struct state_machine_invalid_state {
    static constexpr state_machine_invalid_label_type value = {};
};

template<class StateOrStateMachine>
struct state_machine_terminal_state {
    static constexpr auto value
        = state_machine_invalid_state<StateOrStateMachine>::value;
};

template<class StateInputs, class StateLabel>
class state_definition {
public:
    using transition_type = state_transition<StateInputs, StateLabel>;
    using transition_list = std::vector<std::unique_ptr<transition_type>>;
    using inputs_type     = StateInputs;
    using label_type      = StateLabel;

    state_definition(const state_definition&) = delete;
    auto operator=(const state_definition&) -> state_definition& = delete;
    state_definition(state_definition&&) noexcept                = delete;
    auto operator=(state_definition&&) noexcept -> state_definition& = delete;

    static constexpr auto invalid_state
        = state_machine_invalid_state<state_definition>::value;
    static constexpr auto terminal_state
        = state_machine_terminal_state<state_definition>::value;

    static_assert(
        !std::is_same_v<
            decltype(invalid_state),
            state_machine_invalid_label_type>,
        "State labels must have a value representing invalid state, please "
        "specialize state_machine_invalid_state with this value"
    );

    static_assert(
        !std::is_same_v<
            decltype(terminal_state),
            state_machine_invalid_label_type>,
        "State labels must have a value representing terminal state, although "
        "default template is provided that should be valid if "
        "state_machine_terminal_state is valid, so if seeing this error, check "
        "that first."
    );

    static auto make_initial_state() -> std::unique_ptr<state_definition> {
        return std::make_unique<state_definition>();
    }

    static auto make_invalid_state() -> std::unique_ptr<state_definition> {
        return std::make_unique<invalid_state_t>();
    }

    virtual auto possible_transitions() const -> const transition_list& = 0;
    virtual auto label() const -> const label_type&                     = 0;
    virtual ~state_definition() = default;

  protected:
    state_definition() = default;

private:
    class invalid_state_t;
};

template<class StateInputs, class StateLabel>
class state_definition<StateInputs, StateLabel>::invalid_state_t final : public state_definition {
public:
    auto possible_transitions() const -> const transition_list& override {
        static constexpr transition_list empty_list;
        return empty_list;
    }
    auto label() const -> const label_type& override {
        return state_machine_invalid_state<state_definition>::value;
    }
};

template<class StateMachine>
struct state_machine_traits {
    using state_type      = typename StateMachine::state_type;
    using transition_type = typename state_type::transition_type;
    using transition_list = typename state_type::transition_list;
    using label_type      = typename state_type::label_type;
    using inputs_type     = typename state_type::inputs_type;
    static constexpr auto invalid_state
        = state_machine_invalid_state<StateMachine>::value;
    static constexpr auto terminal_state
        = state_machine_terminal_state<StateMachine>::value;
};

template <class>
struct is_state_definition : std::false_type {};

template <class StateInputs, class StateLabel>
struct is_state_definition<state_definition<StateInputs, StateLabel>> : std::true_type {};

template <class StateInputs, class StateLabel, template <class, class> class StateManager>
struct is_state_definition<StateManager<StateInputs, StateLabel>> {
    static constexpr bool value = std::is_base_of_v<
        state_definition<StateInputs, StateLabel>,
        StateManager<StateInputs, StateLabel>>;
};

template <class T>
static constexpr auto is_state_definition_v = is_state_definition<T>::value;

template <class T>
concept StateDefinitionType = is_state_definition_v<T>;

struct default_state_machine_tag {};

template<class StateDefinition>
requires StateDefinitionType<StateDefinition>
class state_machine {
  public:
    using state_type = StateDefinition;
    using inputs_type
        = typename state_machine_traits<state_machine>::inputs_type;

    void transition_states(const inputs_type& inputs) {
        static constexpr auto invalid_label_value
            = state_machine_traits<state_machine>::invalid_state;
        static constexpr auto terminal_label_value
            = state_machine_traits<state_machine>::terminal_state;
        auto new_state = state_type::make_invalid_state();

        if (current_state_->label() == terminal_label_value) {
            current_state_ = std::move(new_state);
            return;
        }

        for (const auto& transition : current_state_->possible_transitions()) {
            if (auto transition_result
                = transition->execute(inputs, *current_state_);
                transition_result->label() != invalid_label_value) {
                new_state = new_state->label() == invalid_label_value
                              ? std::move(transition_result)
                              : throw std::runtime_error(
                                    "Multiple transitions possible, which is "
                                    "not allowed"
                                );
            }
        }
        current_state_ = std::move(new_state);
    }

    auto current_state() const -> const
        typename state_machine_traits<state_machine>::label_type& {
        return current_state_->label();
    }

  private:
    std::unique_ptr<state_type> current_state_{state_type::make_initial_state()};
};

}  // namespace sclx::detail
