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

#include <stdexcept>
#include <utility>
#include <memory>
#include <mutex>
#include <scalix/generic_task.hpp>
#include <scalix/detail/generic_task.hpp>
#include <scalix/scalix_export.hpp>

namespace sclx {

generic_task::generic_task(std::unique_ptr<impl>&& impl)
    : impl_{std::move(impl)},
      metadata_{std::make_shared<task_metadata>()} {}

generic_task::generic_task(
    std::shared_ptr<impl> impl,
    std::shared_ptr<task_metadata> metadata
)
    : impl_{std::move(impl)},
      metadata_{std::move(metadata)} {}

SCALIX_EXPORT void
generic_task::add_dependent_task(const generic_task& dependent_task) const {
    const std::lock_guard lock(metadata_->mutex);
    if (dependent_task.metadata_ == nullptr) {
        throw std::runtime_error("task instance is in an invalid state");
    }
    const std::lock_guard dependent_lock(dependent_task.metadata_->mutex);
    if (metadata_->has_completed) {
        return;
    }

    generic_task copied_task{dependent_task.impl_, dependent_task.metadata_};

    metadata_->dependent_tasks.push_back(std::move(copied_task));

    dependent_task.metadata_->dependency_count++;
}

SCALIX_EXPORT void generic_task::launch() {
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

void generic_task::impl::decrease_dependency_count(
    const std::shared_ptr<task_metadata>& metadata
) const {
    const std::lock_guard lock(metadata->mutex);
    metadata->dependency_count--;
    if (metadata->dependency_count == 0 && metadata->has_launched) {
        this->async_execute(metadata);
    }
}

}  // namespace sclx
