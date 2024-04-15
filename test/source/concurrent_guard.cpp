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

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <future>
#include <mutex>
#include <scalix/concurrent_guard.hpp>
#include <vector>

namespace concurrent_guard_test {
enum class task_tag : std::uint8_t { A, B, C };
}
using namespace concurrent_guard_test;  // NOLINT(*-build-using-namespace)
TEST_CASE("concurrent_guard") {  // NOLINT(*-function-cognitive-complexity)
    sclx::concurrent_guard<int> guard;
    std::promise<void> A2_promise;
    auto A2_future = A2_promise.get_future();
    std::promise<void> B_promise;
    auto B_future = B_promise.get_future();
    std::promise<void> C_promise;
    auto C_future = C_promise.get_future();

    std::mutex order_mutex;
    std::vector<task_tag> order;
    auto fut1 = std::async(std::launch::async, [&] {
        const auto view     = guard.get_view<sclx::access_mode::write>();
        view.access() = 1;
        order.push_back(task_tag::A);
    });

    fut1.wait();

    auto main_thread_view = guard.get_view();
    REQUIRE((main_thread_view.access() == 1));

    main_thread_view.unlock();

    auto fut2 = std::async(std::launch::async, [&] {
        auto view = guard.get_view();
        B_promise.set_value();
        A2_future.wait();
        const std::lock_guard lock(order_mutex);
        order.push_back(task_tag::B);
    });

    auto fut3 = std::async(std::launch::async, [&] {
        B_future.wait();
        C_future.wait();
        const auto view     = guard.get_view<sclx::access_mode::write>();
        view.access() = 2;
        order.push_back(task_tag::A);
    });

    auto fut4 = std::async(std::launch::async, [&] {
        auto view = guard.get_view();
        C_promise.set_value();
        A2_future.wait();
        const std::lock_guard lock(order_mutex);
        order.push_back(task_tag::C);
    });

    B_future.wait();
    C_future.wait();
    A2_promise.set_value();

    fut1.get();
    fut2.get();
    fut3.get();
    fut4.get();

    REQUIRE((order[0] == task_tag::A));
    REQUIRE((order[1] == task_tag::B || order[1] == task_tag::C));
    REQUIRE((order[2] == task_tag::B || order[2] == task_tag::C));
    REQUIRE((order[3] == task_tag::A));

    main_thread_view.lock();
    REQUIRE((main_thread_view.access() == 2));
}