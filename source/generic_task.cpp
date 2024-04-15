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

#include <memory>
#include <scalix/concurrent_guard.hpp>
#include <scalix/detail/generic_task.hpp>
#include <scalix/generic_task.hpp>
#include <stdexcept>
#include <utility>

namespace sclx {

generic_task::generic_task(std::unique_ptr<impl> impl)
    : impl_{std::move(impl)} {}

void generic_task::add_dependent_task(const generic_task& dependent_task) {
    const auto metadata = impl_->metadata_.get_view<access_mode::write>();
    const auto dependent_metadata
        = dependent_task.impl_->metadata_.get_view<access_mode::write>();
    if (metadata.access().has_completed) {
        return;
    }

    metadata.access().dependent_tasks.push_back(dependent_task);

    dependent_metadata.access().dependency_count++;
}

void generic_task::launch() {
    {
        const auto metadata = impl_->metadata_.get_view<access_mode::write>();

        if (metadata.access().has_launched) {
            throw std::runtime_error{"Task has already been launched"};
        }

        metadata.access().has_launched = true;

        if (metadata.access().dependency_count > 0) {
            return;
        }
    }

    impl_->async_execute();
}

generic_task::impl::~impl() = default;

void generic_task::impl::decrease_dependency_count() const {
    auto metadata = metadata_.get_view<access_mode::write>();
    metadata.access().dependency_count--;
    if (metadata.access().dependency_count == 0
        && metadata.access().has_launched) {
        metadata.unlock();
        this->async_execute();
    }
}

}  // namespace sclx
