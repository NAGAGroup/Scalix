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

#include <scalix/concurrent_guard.hpp>
#include <scalix/generic_task.hpp>
#include <vector>

namespace sclx {

class generic_task::impl {
  public:
    struct task_metadata {
        int dependency_count{0};    // cppcheck-suppress unusedStructMember
        bool has_launched{false};   // cppcheck-suppress unusedStructMember
        bool has_completed{false};  // cppcheck-suppress unusedStructMember
        std::vector<generic_task>
            dependent_tasks;  // cppcheck-suppress unusedStructMember
    };

    concurrent_guard<task_metadata> metadata_;

    impl() : metadata_{task_metadata{}} {}

    impl(impl&&) noexcept                    = default;
    auto operator=(impl&&) noexcept -> impl& = default;
    impl(const impl&)                        = delete;
    auto operator=(const impl&) -> impl&     = delete;

    void decrease_dependency_count() const;

    [[nodiscard]] auto has_completed() const -> bool;

    virtual void async_execute() const = 0;

    virtual ~impl();
};

}  // namespace sclx
