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
// #include <sycl/sycl.hpp>

#include <cpptrace/cpptrace.hpp>

namespace sclx {

template<class R>
class typed_task;

class generic_task {
  public:
    generic_task(const generic_task&)                    = delete;
    auto operator=(const generic_task&) -> generic_task& = delete;

    generic_task(generic_task&&)                    = default;
    auto operator=(generic_task&&) -> generic_task& = default;

    void launch();

    void add_dependent_task(const generic_task& dependent_task) const;

    ~generic_task() = default;

    template<class F, class... Args>
    static auto create(F&& func, Args&&... args)
        -> typed_task<std::invoke_result_t<F, Args...>>;

  protected:
    struct task_metadata;
    class impl;

    template<class>
    class specification;

    explicit generic_task(std::unique_ptr<impl> &&impl);
    generic_task(
        std::shared_ptr<impl> impl,
        std::shared_ptr<task_metadata> metadata
    );

    std::shared_ptr<impl> impl_;
    std::shared_ptr<task_metadata> metadata_;
};

struct generic_task::task_metadata {
    int dependency_count{0};
    bool has_launched{false};
    bool has_completed{false};
    std::vector<generic_task> dependent_tasks;
    std::mutex mutex;
};

class generic_task::impl {
    friend class generic_task;

  public:
    impl(impl&&) noexcept                    = default;
    auto operator=(impl&&) noexcept -> impl& = default;
    impl(const impl&)                        = delete;
    auto operator=(const impl&) -> impl&     = delete;

    virtual ~impl();

  protected:
    impl() = default;

    virtual void async_execute(std::shared_ptr<task_metadata> metadata) const
        = 0;

    void decrease_dependency_count(std::shared_ptr<task_metadata> metadata
    ) const {
        const std::lock_guard lock(metadata->mutex);
        metadata->dependency_count--;
        if (metadata->dependency_count == 0 && metadata->has_launched) {
            this->async_execute(metadata);
        }
    }
};

generic_task::generic_task(std::unique_ptr<impl>&& impl)
    : impl_{std::move(impl)},
      metadata_{std::make_shared<task_metadata>()} {}

generic_task::generic_task(
    std::shared_ptr<impl> impl,
    std::shared_ptr<task_metadata> metadata
)
    : impl_{std::move(impl)},
      metadata_{std::move(metadata)} {}

void generic_task::add_dependent_task(const generic_task& dependent_task
) const {
    std::lock_guard lock(metadata_->mutex);
    if (dependent_task.metadata_ == nullptr) {
        throw std::runtime_error("task instance is in an invalid state");
    }
    std::lock_guard dependent_lock(dependent_task.metadata_->mutex);
    if (metadata_->has_completed) {
        return;
    }

    generic_task copied_task{dependent_task.impl_, dependent_task.metadata_};

    metadata_->dependent_tasks.push_back(std::move(copied_task));

    dependent_task.metadata_->dependency_count++;
}

void generic_task::launch() {
    if (metadata_ == nullptr) {
        throw std::runtime_error("task instance is in an invalid state");
    }
    const std::lock_guard lock(metadata_->mutex);
    metadata_->has_launched = true;

    if (metadata_->dependency_count > 0) {
        return;
    }
    impl_->async_execute(std::move(metadata_));
}

generic_task::impl::~impl() = default;

template<class>
class generic_task::specification {};

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

template<class F, class... Args>
auto generic_task::create(F&& func, Args&&... args)
    -> typed_task<std::invoke_result_t<F, Args...>> {
    return typed_task<std::invoke_result_t<F, Args&&...>>{std::make_unique<
        specification<std::invoke_result_t<F, Args&&...>(Args&&...)>>(
        std::forward<F>(func),
        std::forward<Args>(args)...
    )};
}

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
          num_move_assignment_calls(other.num_move_assignment_calls) {
    }

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
    auto task = sclx::generic_task::create(
        pass_argument_rvalue_check,
        std::move(arg)
    );
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
        auto scoped_task = sclx::generic_task::create(
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
            auto scoped_task = sclx::generic_task::create(
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
            auto scoped_task = sclx::generic_task::create(
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
            auto scoped_task = sclx::generic_task::create(
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
            auto scoped_task = sclx::generic_task::create(
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
TEST_CASE(
    "sclx::task dependencies",
    "[sclx::task]"
) {
    std::vector<task_tag> task_order;
    std::mutex task_order_mutex;

    auto task_A = sclx::generic_task::create([&task_order, &task_order_mutex] {
        {
            const std::lock_guard lock(task_order_mutex);
            task_order.push_back(task_tag::A);
        }
    });

    auto task_B = sclx::generic_task::create([&task_order, &task_order_mutex] {
        {
            const std::lock_guard lock(task_order_mutex);
            task_order.push_back(task_tag::B);
        }
    });

    auto task_C = sclx::generic_task::create([&task_order, &task_order_mutex] {
        {
            const std::lock_guard lock(task_order_mutex);
            task_order.push_back(task_tag::C);
        }
    });

    auto task_D = sclx::generic_task::create([&task_order, &task_order_mutex] {
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