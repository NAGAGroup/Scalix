//------------------------------------------------------------------------------
// BSD 3-Clause License
//
// Copyright (c) 2023 Jack Myers
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
//------------------------------------------------------------------------------

#pragma once

#include <scalix/defines.hpp>
#include <scalix/detail/page_handle.hpp>
#include <scalix/pointers.hpp>
#include <unordered_map>
#include <vector>

namespace sclx::detail {

template<template<page_size_t> class AllocationType, page_size_t PageSize>
struct allocation_methods {
    static_assert(false, "No allocation methods defined for this type");

    template<class T, class... Args>
    static std::vector<page_handle<page_handle_type::strong, PageSize>>
    allocate_pages_and_reuse_if_possible(
        device_id_t device_id,
        const std::vector<page_index_t>& indices,
        const std::vector<page_handle<page_handle_type::weak, PageSize>>&
            weak_handles,
        Args&&... args
    ) {}

    template<class T, class... Args>
    static std::vector<page_handle<page_handle_type::strong, PageSize>>
    allocate_pages_and_reuse_if_possible(
        device_id_t device_id,
        page_index_t first,
        page_index_t last,
        const std::vector<page_handle<page_handle_type::weak, PageSize>>&
            weak_handles,
        Args&&... args
    ) {}
};

template<page_size_t PageSize>
class allocation {
  public:
    virtual ~allocation() = default;
};

template<class T, page_size_t PageSize = default_page_size>
class allocation_factory {
  public:
    template<template<page_size_t> class AllocationType, class... Args>
    void allocate_pages_and_reuse_if_possible(
        device_id_t device_id,
        const std::vector<page_index_t>& indices,
        const std::vector<page_handle<page_handle_type::weak, PageSize>>&
            weak_handles,
        Args&&... args
    ) {
        auto pages
            = allocation_methods<AllocationType, PageSize>::
                template allocate_pages_and_reuse_if_possible<T>(
                device_id,
                indices,
                weak_handles,
                std::forward<Args>(args)...
            );
        pages_.insert(
            pages_.end(),
            std::make_move_iterator(pages.begin()),
            std::make_move_iterator(pages.end())
        );
    }

    template<template<page_size_t> class AllocationType, class... Args>
    void allocate_pages_and_reuse_if_possible(
        device_id_t device_id,
        page_index_t first,
        page_index_t last,
        const std::vector<page_handle<page_handle_type::weak, PageSize>>&
            weak_handles,
        Args&&... args
    ) {
        auto pages = allocation_methods<AllocationType, PageSize>::
            template allocate_pages_and_reuse_if_possible<T>(
                device_id,
                first,
                last,
                weak_handles,
                std::forward<Args>(args)...
            );
        pages_.insert(
            pages_.end(),
            std::make_move_iterator(pages.begin()),
            std::make_move_iterator(pages.end())
        );
    }

    const std::vector<page_handle<page_handle_type::strong, PageSize>>&
    pages() const {
        return pages_;
    }

  private:
    std::vector<page_handle<page_handle_type::strong, PageSize>> pages_;
};

}  // namespace sclx::detail
