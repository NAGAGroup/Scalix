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
#include <future>
#include <memory>
#include <algorithm>
#include <scalix/concurrent_guard.hpp>
#include <scalix/defines.hpp>
#include <utility>

namespace sclx::detail {

template<page_size_t>
class page_data_interface {
  public:
    page_data_interface() = default;

    page_data_interface(const page_data_interface&) = default;
    page_data_interface(page_data_interface&&)      = default;

    auto operator=(const page_data_interface&)
        -> page_data_interface&                                   = default;
    auto operator=(page_data_interface&&) -> page_data_interface& = default;

    virtual auto copy_to(page_data_interface& other) const -> std::future<void>
        = 0;
    virtual auto copy_to(page_ptr_t destination) const -> std::future<void> = 0;

    virtual auto copy_from(concurrent_view<const page_ptr_t>&& source)
        -> std::future<void>
        = 0;

    [[nodiscard]] virtual auto page_address() const -> const byte* = 0;

    [[nodiscard]] virtual auto check_if_same_page(page_ptr_t other) const
        -> bool
        = 0;

    [[nodiscard]] virtual auto operator==(const page_data_interface& other
    ) const -> bool
        = 0;

    [[nodiscard]] virtual auto operator!=(const page_data_interface& other
    ) const -> bool
        = 0;

    virtual ~page_data_interface() = default;
};

template<page_size_t PageSize>
class page_data final : public page_data_interface<PageSize> {
  public:
    using alloc_handle_t            = std::shared_ptr<void>;
    static constexpr auto page_size = PageSize;

    page_data() = default;

    // ReSharper disable once CppParameterMayBeConst
    page_data(page_ptr_t data, alloc_handle_t alloc_handle)
        : data_{data},
          alloc_handle_{std::move(alloc_handle)} {}

    auto copy_to(page_data_interface<page_size>& other) const
        -> std::future<void> override {
        return other.copy_from(data_.get_view<access_mode::read>());
    }

    auto copy_to(page_ptr_t destination) const -> std::future<void> override {
        return std::async([*this, destination] {
            if (!data_.valid() || destination == nullptr
                || check_if_same_page(destination)) {
                return;
            }
            const auto source = data_.get_view<access_mode::read>();
            std::copy_n(source.access(), page_size, destination);
        });
    }

    auto copy_from(concurrent_view<const page_ptr_t>&& source)
        -> std::future<void> override {
        return std::async([*this, source = std::move(source)] {
            if (source.access() == nullptr || !data_.valid()
                || check_if_same_page(source.access())) {
                return;
            }
            const auto data = data_.get_view<access_mode::write>();
            std::copy_n(source.access(), page_size, data.access());
        });
    }

    [[nodiscard]] auto page_address() const -> const byte* override {
        return data_.unsafe_access();
    }

    // ReSharper disable once CppParameterMayBeConst
    [[nodiscard]] auto check_if_same_page(page_ptr_t other) const
        -> bool override {
        if (!data_.valid() || other == nullptr) {
            return false;
        }
        return data_.unsafe_access() == other;
    }

    [[nodiscard]] auto operator==(const page_data_interface<page_size>& other
    ) const -> bool override {
        return other.check_if_same_page(data_.unsafe_access());
    }

    [[nodiscard]] auto operator!=(const page_data_interface<page_size>& other
    ) const -> bool override {
        return !(*this == other);
    }

  private:
    concurrent_guard<page_ptr_t> data_{nullptr};
    alloc_handle_t alloc_handle_;
};

}  // namespace sclx::detail
