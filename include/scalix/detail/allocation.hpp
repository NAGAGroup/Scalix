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

#pragma once

#include <scalix/defines.hpp>
#include <scalix/detail/page_handle.hpp>
#include <scalix/pointers.hpp>
#include <unordered_map>
#include <vector>

namespace sclx::detail {

enum class reuse_pages : std::uint8_t { if_possible, no };

struct allocation_handle_base {
    allocation_handle_base() = default;

    allocation_handle_base(const allocation_handle_base&) = delete;
    auto operator=(const allocation_handle_base&)
        -> allocation_handle_base& = delete;

    allocation_handle_base(allocation_handle_base&&) = default;
    auto
    operator=(allocation_handle_base&&) -> allocation_handle_base& = default;

    virtual ~allocation_handle_base() = default;
};

template<page_size_t PageSize>
class allocation_handle : public allocation_handle_base {
  public:
    using page_handle = page_handle<page_handle_type::strong, PageSize>;
    using allocation_handle_base::allocation_handle_base;

    [[nodiscard]] virtual auto device_id() const -> device_id_t = 0;
    [[nodiscard]] virtual auto
    pages() const -> const std::vector<page_handle>& = 0;
};

struct access_anchor_interface;

class access_anchor {
  public:
    access_anchor()                                        = default;
    access_anchor(access_anchor&)                          = default;
    access_anchor(access_anchor&&)                         = default;
    auto operator=(const access_anchor&) -> access_anchor& = default;
    auto operator=(access_anchor&&) -> access_anchor&      = default;

    ~access_anchor() = default;

  private:
    access_anchor(std::shared_ptr<allocation_handle_base> handle)
        : handle_(std::move(handle)) {}

    template<page_size_t PageSize>
    [[nodiscard]] auto get_allocation() -> allocation_handle<PageSize>& {
        return static_cast<allocation_handle<PageSize>&>(*handle_);
    }

    template<page_size_t PageSize>
    [[nodiscard]] auto
    get_allocation() const -> const allocation_handle<PageSize>& {
        return static_cast<allocation_handle<PageSize>&>(*handle_);
    }

    std::shared_ptr<allocation_handle_base> handle_;

    friend struct access_anchor_interface;
};

struct access_anchor_interface {
    template<class AllocationHandle>
    static auto create_anchor() -> access_anchor {
        return {std::make_shared<AllocationHandle>()};
    }

    template<page_size_t PageSize>
    static auto get_allocation(access_anchor& anchor
    ) -> allocation_handle<PageSize>& {
        return anchor.get_allocation<PageSize>();
    }

    template<page_size_t PageSize>
    static auto get_allocation(const access_anchor& anchor
    ) -> const allocation_handle<PageSize>& {
        return anchor.get_allocation<PageSize>();
    }
};

template<
    class T,
    template<class, reuse_pages, page_size_t>
    class AllocationType,
    reuse_pages ReusePagesFlag,
    page_size_t PageSize,
    class... Args>
concept AllocationConcept = requires(
    AllocationType<T, ReusePagesFlag, PageSize> alloc,
    Args&&... args
) {
    {
        AllocationType<T, ReusePagesFlag, PageSize>(
            device_id_t(),
            std::declval<const std::vector<page_index_t>&>(),
            std::declval<const std::vector<
                page_handle<page_handle_type::weak, PageSize>>&>(),
            std::forward<Args>(args)...
        )
    };
    {
        AllocationType<T, ReusePagesFlag, PageSize>(
            device_id_t(),
            page_index_t(),
            page_index_t(),
            std::declval<const std::vector<
                page_handle<page_handle_type::weak, PageSize>>&>(),
            std::forward<Args>(args)...
        )
    };
    {
        static_cast<const AllocationType<T, ReusePagesFlag, PageSize>&>(alloc)
            .anchor()
    } -> std::same_as<const access_anchor&>;
};

}  // namespace sclx::detail

namespace sclx {

using access_anchor = detail::access_anchor;

}
