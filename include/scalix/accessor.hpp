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
#include <cstdint>
#include <scalix/defines.hpp>
#include <type_traits>

namespace sclx {

namespace detail {

template<int Dimensions, int Iter = 0>
size_t linearize_id(
    const id<Dimensions>& index,
    const range<Dimensions>& range,
    size_t multiplier = 1
) {
    if constexpr (Iter == Dimensions - 1) {
        return 0;
    }

    return linearize_id<Dimensions, Iter + 1>(
               index,
               range,
               multiplier * range[Dimensions - Iter - 1]
           )
         + index[Dimensions - Iter - 1] * multiplier;
}

template<class T>
constexpr page_size_t utilized_bytes_per_page(const page_size_t page_size) {
    return page_size / sizeof(T) * sizeof(T);
}

static constexpr page_index_t bad_page = -1;

template<class T, class IndexType>
    requires std::is_integral_v<IndexType>
page_index_t
map_index_to_page(const page_size_t page_size, const IndexType index) {
    if constexpr (std::is_signed_v<IndexType>) {
        if (index < 0) {
            return bad_page;
        }
    }

    const auto byte_index     = sizeof(T) * index;
    const auto utilized_bytes = utilized_bytes_per_page<T>(page_size);
    if (byte_index / utilized_bytes
        > std::numeric_limits<page_index_t>::max()) {
        return bad_page;
    }

    return static_cast<page_index_t>(byte_index / utilized_bytes);
}

template<class T, class IndexType>
    requires std::is_integral_v<IndexType>
byte_offset
map_index_to_byte_offset(const page_size_t page_size, const IndexType index) {
    const auto byte_index     = sizeof(T) * index;
    const auto utilized_bytes = utilized_bytes_per_page<T>(page_size);
    return static_cast<byte_offset>(byte_index % utilized_bytes);
}

}  // namespace detail

template<class, int>
class buffer;

template<
    class T,
    int Dimensions,
    access_mode AccessMode
    = std::is_const_v<T> ? access_mode::read : access_mode::read_write>
class accessor {
  public:
    using value_type = T;
    using reference_type
        = std::conditional_t<AccessMode == access_mode::read, const T&, T&>;

    friend class buffer<T, Dimensions>;

    accessor(const accessor&) = default;
    accessor(accessor&&)      = default;

    accessor& operator=(const accessor&) = default;
    accessor& operator=(accessor&&)      = default;

    [[nodiscard]] auto operator[](const size_t index) const
        -> std::enable_if_t<(Dimensions > 1), reference_type> {
        static T bad_value{};

        const auto page_index = detail::map_index_to_page<T>(page_size_, index);
        if (page_index == detail::bad_page) {
            return bad_value;
        }

        const auto byte_offset
            = detail::map_index_to_byte_offset<T>(page_size_, index);
        value_type* page = reinterpret_cast<value_type*>(
            pages_[page_index] + byte_offset
        );  // NOLINT(*-pro-type-reinterpret-cast)
        return static_cast<reference_type>(*page);
    }

    [[nodiscard]] auto operator[](id<Dimensions> index) const
        -> reference_type {
        return this->operator[](detail::linearize_id(index, range_));
    }

    ~accessor() = default;

  private:
    const page_ptr_t* pages_;
    range<Dimensions> range_;
    page_size_t page_size_;

    accessor(
        const page_ptr_t* pages,
        const range<Dimensions>& range,
        const page_size_t page_size
    )
        : pages_(pages),
          range_(range),
          page_size_(page_size) {}
};

}  // namespace sclx
