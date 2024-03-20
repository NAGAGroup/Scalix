// BSD 3-Clause License
//
// Copyright (c) 2023-2024 Jack Myers
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

#include <catch2/catch_test_macros.hpp>
#include <future>
#include <memory>
#include <scalix/typed_task.hpp>

namespace sclx {

template<class T>
struct remove_rvalue_reference {
    using type = T;
};

template<class T>
struct remove_rvalue_reference<T&&> {
    using type = T;
};

}  // namespace sclx

struct argument_rvalue_check {
    argument_rvalue_check() = default;

    argument_rvalue_check(const argument_rvalue_check& /*unused*/) {}
    auto operator=(const argument_rvalue_check& /*unused*/)
        -> argument_rvalue_check& {
        constructed_via_rvalue = false;
        return *this;
    }

    argument_rvalue_check(argument_rvalue_check&& /*unused*/) noexcept
        : constructed_via_rvalue{true} {}
    auto operator=(argument_rvalue_check&& /*unused*/) noexcept
        -> argument_rvalue_check& {
        constructed_via_rvalue = true;
        return *this;
    }

    ~argument_rvalue_check() = default;

    bool constructed_via_rvalue{false};
};

auto pass_argument_rvalue_check(argument_rvalue_check arg)
    -> argument_rvalue_check {
    return arg;
}

struct constructor_assignment_counter {
    constructor_assignment_counter() = default;

    constructor_assignment_counter(const constructor_assignment_counter& other)
        : num_copy_constructor_calls(other.num_copy_constructor_calls + 1),
          num_move_constructor_calls(other.num_move_constructor_calls),
          num_copy_assignment_calls(other.num_copy_assignment_calls),
          num_move_assignment_calls(other.num_move_assignment_calls) {}

    auto operator=(const constructor_assignment_counter& other)
        -> constructor_assignment_counter& {
        num_copy_constructor_calls = other.num_copy_constructor_calls;
        num_copy_assignment_calls  = other.num_copy_assignment_calls + 1;
        num_move_constructor_calls = other.num_move_constructor_calls;
        num_move_assignment_calls  = other.num_move_assignment_calls;

        return *this;
    }

    constructor_assignment_counter(constructor_assignment_counter&& other
    ) noexcept
        : num_copy_constructor_calls(other.num_copy_constructor_calls),
          num_move_constructor_calls(other.num_move_constructor_calls + 1),
          num_copy_assignment_calls(other.num_copy_assignment_calls),
          num_move_assignment_calls(other.num_move_assignment_calls) {}

    auto operator=(constructor_assignment_counter&& other) noexcept
        -> constructor_assignment_counter& {
        num_copy_constructor_calls = other.num_copy_constructor_calls;
        num_copy_assignment_calls  = other.num_copy_assignment_calls;
        num_move_constructor_calls = other.num_move_constructor_calls;
        num_move_assignment_calls  = other.num_move_assignment_calls + 1;

        return *this;
    }

    auto assign_no_count(const constructor_assignment_counter& other)
        -> constructor_assignment_counter& {
        num_copy_constructor_calls = other.num_copy_constructor_calls;
        num_copy_assignment_calls  = other.num_copy_assignment_calls;
        num_move_constructor_calls = other.num_move_constructor_calls;
        num_move_assignment_calls  = other.num_move_assignment_calls;

        return *this;
    }

    auto assign_no_count(constructor_assignment_counter&& other)
        -> constructor_assignment_counter& {
        num_copy_constructor_calls = other.num_copy_constructor_calls;
        num_copy_assignment_calls  = other.num_copy_assignment_calls;
        num_move_constructor_calls = other.num_move_constructor_calls;
        num_move_assignment_calls  = other.num_move_assignment_calls;

        return *this;
    }

    int num_copy_constructor_calls{0};
    int num_move_constructor_calls{0};
    int num_copy_assignment_calls{0};
    int num_move_assignment_calls{0};
};

TEST_CASE(
    "sclx::task arg behavior",
    "[sclx::task]"
) {  // NOLINT(*-function-cognitive-complexity)
    argument_rvalue_check arg;
    auto task = sclx::create_task(pass_argument_rvalue_check, std::move(arg));
    auto task_future = task.get_future();
    task.launch();
    auto result = task_future.get();
    REQUIRE(result.constructed_via_rvalue);

    std::promise<void> promise;
    auto fut                  = promise.get_future();
    using correct_use_count_t = bool;
    using valid_weak_ptr_t    = bool;
    using correct_data_t      = bool;
    std::shared_future<
        std::tuple<correct_use_count_t, valid_weak_ptr_t, correct_data_t>>
        task_future2;
    {
        constexpr int expected_value = 42;

        auto int_ptr = std::make_shared<int>(expected_value);
        std::weak_ptr int_wptr{int_ptr};
        auto scoped_task = sclx::create_task(
            [&fut](std::shared_ptr<int>&& moved_ptr, std::weak_ptr<int>& wptr) {
                fut.wait();
                const auto ptr               = std::move(moved_ptr);
                const bool correct_use_count = ptr.use_count() == 1;
                const bool valid_weak_ptr
                    = !wptr.expired() && (wptr.lock().get() == ptr.get());
                const bool correct_data = *ptr == expected_value;
                return std::make_tuple(
                    correct_use_count,
                    valid_weak_ptr,
                    correct_data
                );
            },
            std::move(int_ptr),
            int_wptr
        );
        task_future2 = scoped_task.get_future().share();
        scoped_task.launch();
    }
    promise.set_value();
    auto [correct_use_count, valid_weak_ptr, correct_data] = task_future2.get();
    REQUIRE(correct_use_count);
    REQUIRE(valid_weak_ptr);
    REQUIRE(correct_data);

    {
        constructor_assignment_counter counter;
        {
            auto scoped_task = sclx::create_task(
                [](constructor_assignment_counter&& counter
                ) -> std::unique_ptr<constructor_assignment_counter> {
                    return std::make_unique<constructor_assignment_counter>(
                        std::move(counter)
                    );
                },
                std::move(counter)
            );
            scoped_task.launch();
            counter.assign_no_count(std::move(*scoped_task.get_future().get()));
        }
        REQUIRE(counter.num_copy_constructor_calls == 0);
        REQUIRE(
            counter.num_move_constructor_calls == 2
        );  // one for task args tuple, one for move from tuple to lambda to
            // pointer
        REQUIRE(counter.num_copy_assignment_calls == 0);
        REQUIRE(counter.num_move_assignment_calls == 0);

        {
            auto scoped_task = sclx::create_task(
                [](constructor_assignment_counter counter
                ) -> std::unique_ptr<constructor_assignment_counter> {
                    auto ptr
                        = std::make_unique<constructor_assignment_counter>();
                    ptr->assign_no_count(counter);
                    return ptr;
                },
                std::move(counter)
            );
            scoped_task.launch();
            counter.assign_no_count(*scoped_task.get_future().get());
        }
        REQUIRE(counter.num_copy_constructor_calls == 0);
        REQUIRE(
            counter.num_move_constructor_calls == 4
        );  // one for task args tuple, one for move from tuple to lambda
        REQUIRE(counter.num_copy_assignment_calls == 0);
        REQUIRE(counter.num_move_assignment_calls == 0);

        {
            auto scoped_task = sclx::create_task(
                [](const constructor_assignment_counter& counter
                ) -> std::unique_ptr<constructor_assignment_counter> {
                    auto ptr
                        = std::make_unique<constructor_assignment_counter>();
                    ptr->assign_no_count(counter);
                    return ptr;
                },
                counter
            );
            scoped_task.launch();
            counter.assign_no_count(*scoped_task.get_future().get());
        }
        REQUIRE(counter.num_copy_constructor_calls == 1);
        REQUIRE(
            counter.num_move_constructor_calls == 4
        );  // one for task args tuple, one for move from tuple to lambda
        REQUIRE(counter.num_copy_assignment_calls == 0);
        REQUIRE(counter.num_move_assignment_calls == 0);

        {
            auto scoped_task = sclx::create_task(
                [](constructor_assignment_counter counter
                ) -> std::unique_ptr<constructor_assignment_counter> {
                    auto ptr
                        = std::make_unique<constructor_assignment_counter>();
                    ptr->assign_no_count(counter);
                    return ptr;
                },
                counter
            );
            scoped_task.launch();
            counter.assign_no_count(*scoped_task.get_future().get());
        }
        REQUIRE(counter.num_copy_constructor_calls == 3);
        REQUIRE(
            counter.num_move_constructor_calls == 4
        );  // one for task args tuple, one for move from tuple to lambda
        REQUIRE(counter.num_copy_assignment_calls == 0);
        REQUIRE(counter.num_move_assignment_calls == 0);
    }
}

// this tests the task dependency system
// we test the diamond dependency pattern:
// 1. task A is parent of tasks B and C
// 2. task B and C are parents of task D
enum class task_tag : std::uint8_t { A, B, C, D };
TEST_CASE("sclx::task dependencies", "[sclx::task]") {
    std::vector<task_tag> task_order;
    std::mutex task_order_mutex;

    auto task_A = sclx::create_task([&task_order, &task_order_mutex] {
        {
            const std::lock_guard lock(task_order_mutex);
            task_order.push_back(task_tag::A);
        }
    });

    auto task_B = sclx::create_task([&task_order, &task_order_mutex] {
        {
            const std::lock_guard lock(task_order_mutex);
            task_order.push_back(task_tag::B);
        }
    });

    auto task_C = sclx::create_task([&task_order, &task_order_mutex] {
        {
            const std::lock_guard lock(task_order_mutex);
            task_order.push_back(task_tag::C);
        }
    });

    auto task_D = sclx::create_task([&task_order, &task_order_mutex] {
        {
            const std::lock_guard lock(task_order_mutex);
            task_order.push_back(task_tag::D);
        }
    });

    auto A_fut = task_A.get_future();
    auto B_fut = task_B.get_future();
    auto C_fut = task_C.get_future();
    auto D_fut = task_D.get_future();

    task_A.add_dependent_task(task_B);
    task_A.add_dependent_task(task_C);
    task_B.add_dependent_task(task_D);
    task_C.add_dependent_task(task_D);

    task_D.launch();
    task_B.launch();
    task_C.launch();
    task_A.launch();

    A_fut.get();
    B_fut.get();
    C_fut.get();
    D_fut.get();

    REQUIRE((task_order.size() == 4));
    REQUIRE((task_order[0] == task_tag::A));
    REQUIRE(((task_order[1] == task_tag::B) || (task_order[1] == task_tag::C)));
    REQUIRE(((task_order[2] == task_tag::B) || (task_order[2] == task_tag::C)));
    REQUIRE((task_order[3] == task_tag::D));
}