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
#include <future>
#include <scalix/defines.hpp>
#include <scalix/pointers.hpp>
#include <variant>

namespace sclx::detail {

template<page_size_t PageSize>
class allocation;

template<page_size_t PageSize>
struct page_data {
    sclx::byte* data{nullptr};
    sclx::write_bit_t write_bit{0};
};

template<page_size_t PageSize>
class page_table_interface;

template<page_size_t PageSize = default_page_size>
class page_interface {
  public:
    virtual auto data() -> std::variant<sclx::byte*, std::future<sclx::byte*>>
        = 0;
    virtual auto device_id() -> std::variant<device_id_t, sclx::mpi_device> = 0;
    virtual auto write_bit()
        -> std::variant<sclx::write_bit_t, std::future<write_bit_t>>
        = 0;
    virtual auto set_write_bit(sclx::write_bit_t bit) -> sclx::event = 0;
    virtual auto index() -> page_index_t                             = 0;
    virtual auto copy_from(sycl::queue q, std::shared_ptr<page_interface> src)
        -> sclx::event
        = 0;

    virtual auto is_mpi_local() -> bool { return true; }

    virtual ~page_interface() = default;
};

struct page_copy_rules {
    bool expect_valid_src{true};
    bool expect_valid_dst{true};
    bool ignore_unwritten{false};
    bool ignore_page_index{false};
};

enum class page_handle_type { strong, weak };

template<page_size_t PageSize>
class allocation;

template<page_handle_type HandleType, page_size_t PageSize>
class page_handle;

template<page_handle_type HandleType, page_size_t PageSize>
struct page_handle_traits {
    using page_pointer = std::conditional_t<
        HandleType == page_handle_type::strong,
        std::shared_ptr<page_interface<PageSize>>,
        std::weak_ptr<page_interface<PageSize>>>;
};

template<page_handle_type HandleType, page_size_t PageSize>
struct page_handle_creator_struct;

template<page_size_t PageSize>
class page_handle<page_handle_type::weak, PageSize> {
  public:
    static constexpr auto handle_type = page_handle_type::weak;
    using page_pointer =
        typename page_handle_traits<handle_type, PageSize>::page_pointer;

    page_handle() = default;

    page_handle(const page_handle&)                    = default;
    page_handle(page_handle&&)                         = default;
    auto operator=(const page_handle&) -> page_handle& = default;
    auto operator=(page_handle&&) -> page_handle&      = default;

    explicit page_handle(page_handle<page_handle_type::strong, PageSize> other)
        : impl_(std::move(other.impl_)) {}

    void release() { impl_.reset(); }

    auto lock() const -> page_handle<page_handle_type::strong, PageSize> {
        return page_handle<page_handle_type::strong, PageSize>{impl_.lock()};
    }

    ~page_handle() = default;

    friend class page_handle<page_handle_type::weak, PageSize>;
    friend class page_handle<page_handle_type::strong, PageSize>;

    friend struct page_handle_creator_struct<handle_type, PageSize>;

  private:
    explicit page_handle(page_pointer impl) : impl_(std::move(impl)) {}

    page_pointer impl_;
};

template<page_size_t PageSize>
class page_handle<page_handle_type::strong, PageSize> {
  public:
    static constexpr auto handle_type = page_handle_type::strong;
    using page_pointer =
        typename page_handle_traits<handle_type, PageSize>::page_pointer;

    page_handle() = default;

    page_handle(const page_handle&)                    = default;
    page_handle(page_handle&&)                         = default;
    auto operator=(const page_handle&) -> page_handle& = default;
    auto operator=(page_handle&&) -> page_handle&      = default;

    [[nodiscard]] auto data() const
        -> std::variant<sclx::byte*, std::future<sclx::byte*>> {
        if (!is_valid()) {
            return nullptr;
        }
        return impl_->data();
    }

    [[nodiscard]] auto device_id() const
        -> std::variant<device_id_t, sclx::mpi_device> {
        if (!is_valid()) {
            return no_device;
        }
        return impl_->device_id();
    }

    [[nodiscard]] auto write_bit() const
        -> std::variant<write_bit_t, std::future<write_bit_t>> {
        if (!is_valid()) {
            return write_bit_t{0};
        }
        return impl_->write_bit();
    }

    void set_write_bit(sclx::write_bit_t bit) {
        if (!is_valid()) {
            return;
        }
        impl_->set_write_bit(bit);
    }

    [[nodiscard]] auto index() const -> page_index_t {
        if (!is_valid()) {
            return invalid_page;
        }
        return impl_->index();
    }

    [[nodiscard]] auto is_valid() const -> bool { return impl_ != nullptr; }

    void release() { impl_.reset(); }

    auto copy_from(sycl::queue q, page_handle src, page_copy_rules rules = {})
        -> sclx::event {
        auto skip_copy_variant = should_skip_copy(*this, src, rules);
        if (src.is_mpi_local()) {
            if (std::get<bool>(skip_copy_variant)) {
                return {};
            }
        } else {
            auto skip_copy_future
                = std::get<std::future<bool>>(skip_copy_variant).share();
            auto skip_copy_ptr   = std::make_shared<bool>();
            auto skip_copy_event = q.submit([&](sycl::handler& cgh) {
                cgh.host_task([=]() { *skip_copy_ptr = skip_copy_future.get(); }
                );
            });
            return q.submit([&](sycl::handler& cgh) {
                cgh.depends_on(skip_copy_event);
                cgh.host_task([src, skip_copy_ptr, q, this]() {
                    if (*skip_copy_ptr) {
                        return;
                    }
                    this->impl_->copy_from(q, src.impl_).wait_and_throw();
                });
            });
        }

        return impl_->copy_from(q, src.impl_);
    }

    [[nodiscard]] auto is_mpi_local() const -> bool {
        if (!is_valid()) {
            return true;
        }
        return impl_->is_mpi_local();
    }

    auto replace_with(sycl::queue q, page_handle new_page) -> sclx::event {
        if (!this->is_valid()) {
            *this = std::move(new_page);
            return {};
        }
        if (!new_page.is_valid()) {
            throw std::runtime_error(
                "Cannot replace a page with an invalid page"
            );
        }
        auto& old_page  = *this;
        auto copy_event = new_page.copy_from(
            q,
            old_page,
            page_copy_rules{
                .expect_valid_src = false,
                .expect_valid_dst = false
            }
        );
        if (old_page.is_mpi_local()) {
            auto write_bit_variant = old_page.write_bit();
            auto write_bit         = std::get<write_bit_t>(write_bit_variant);
            new_page.set_write_bit(write_bit);
            *this = std::move(new_page);
            return copy_event;
        }

        auto shared_write_bit          = std::make_shared<write_bit_t>();
        auto write_bit_retrieval_event = q.submit([&](sycl::handler& cgh) {
            cgh.host_task([=]() {
                auto write_bit_variant = old_page.write_bit();
                auto& write_bit_future
                    = std::get<std::future<write_bit_t>>(write_bit_variant);
                *shared_write_bit = write_bit_future.get();
            });
        });
        auto write_bit_event
            = q.submit([&, write_bit_retrieval_event](sycl::handler& cgh) {
                  cgh.depends_on(write_bit_retrieval_event);
                  cgh.host_task([=]() mutable {
                      new_page.set_write_bit(*shared_write_bit);
                  });
              });
        *this = std::move(new_page);
        return q.submit([copy_event, write_bit_event](sycl::handler& cgh) {
            cgh.depends_on(std::move(copy_event));
            cgh.depends_on(std::move(write_bit_event));
        });
    }

    ~page_handle() = default;

    friend class page_handle<page_handle_type::weak, PageSize>;
    friend struct page_handle_creator_struct<handle_type, PageSize>;

  private:
    explicit page_handle(page_pointer impl) : impl_(std::move(impl)) {}

    static auto should_skip_copy(
        const page_handle& dst,
        const page_handle& src,
        page_copy_rules rules
    ) -> std::variant<bool, std::future<bool>> {
        if (rules.expect_valid_dst) {
            if (!dst.is_valid()) {
                throw std::runtime_error(
                    "Destination page is invalid in copy_from"
                );
            }
        } else {
            if (dst.is_valid()) {
                return true;
            }
        }

        if (rules.expect_valid_src) {
            if (!src.is_valid()) {
                throw std::runtime_error("Source page is invalid in copy_from");
            }
        } else {
            if (!src.is_valid()) {
                return true;
            }
        }

        if (rules.ignore_unwritten) {
            auto write_bit_variant = src.write_bit();
            if (src.is_mpi_local()) {
                auto write_bit = std::get<write_bit_t>(write_bit_variant);
                if (write_bit == 0) {
                    return true;
                }
            } else {
                auto& write_bit_future
                    = std::get<std::future<write_bit_t>>(write_bit_variant);
                return std::async(
                    std::launch::async,
                    [write_bit_future = std::move(write_bit_future)]() mutable {
                        return write_bit_future.get() == 0;
                    }
                );
            }
        }

        if (!rules.ignore_page_index) {
            if (dst.index() != src.index()) {
                throw std::runtime_error(
                    "Page indices do not match in copy_from"
                );
            }
        }

        return false;
    }

    page_pointer impl_;
};

template<page_handle_type HandleType, page_size_t PageSize>
struct page_handle_creator_struct {
    static auto
    create(typename page_handle_traits<HandleType, PageSize>::page_pointer impl)
        -> page_handle<HandleType, PageSize> {
        return page_handle<HandleType, PageSize>(impl);
    }
};

template<page_handle_type HandleType, page_size_t PageSize>
auto make_page_handle(
    typename page_handle_traits<HandleType, PageSize>::page_pointer impl
) -> page_handle<HandleType, PageSize> {
    return page_handle_creator_struct<HandleType, PageSize>::create(impl);
}

template<class T, page_size_t PageSize = default_page_size>
struct page_traits {
    static constexpr page_size_t page_size         = PageSize;
    static constexpr page_size_t elements_per_page = page_size / sizeof(T);
    static constexpr page_size_t allocated_bytes_per_page
        = elements_per_page * sizeof(T);
    static_assert(elements_per_page > 0, "Page size is too small for type T");
};

template<class T, page_size_t PageSize = default_page_size>
constexpr auto required_pages_for_elements(page_count_t elements)
    -> page_count_t {
    return (elements + page_traits<T, PageSize>::elements_per_page - 1)
         / page_traits<T, PageSize>::elements_per_page;
}

template<
    class T,
    class PageBeginIterator,
    class PageEndIterator,
    class PageDstIterator,
    page_size_t PageSize = default_page_size>
auto copy_pages(
    const sycl::queue& q,
    PageBeginIterator begin,
    PageEndIterator end,
    PageDstIterator dst,
    page_copy_rules rules = {}
) -> sclx::event {
    std::vector<sclx::event> copy_events(std::distance(begin, end));
    std::transform(
        begin,
        end,
        copy_events.begin(),
        [q, dst, rules](auto& page) { return page.copy_from(*dst++, rules); }
    );

    return q.submit([copy_events = std::move(copy_events)](sycl::handler& cgh) {
        cgh.depends_on(copy_events);
    });
}

}  // namespace sclx::detail
