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
    virtual std::variant<sclx::byte*, std::future<sclx::byte*>> data() = 0;
    virtual std::variant<device_id_t, sclx::mpi_device> device_id()    = 0;
    virtual std::variant<sclx::write_bit_t, std::future<write_bit_t>>
    write_bit()                                              = 0;
    virtual sclx::event set_write_bit(sclx::write_bit_t bit) = 0;
    virtual page_index_t index()                             = 0;
    virtual sclx::event
    copy_from(sycl::queue q, std::shared_ptr<page_interface> src)
        = 0;

    virtual bool is_mpi_local() { return true; }

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
    using alloc_pointer = std::conditional_t<
        HandleType == page_handle_type::strong,
        std::shared_ptr<sclx::detail::allocation<PageSize>>,
        std::weak_ptr<sclx::detail::allocation<PageSize>>>;
    using page_pointer
        = std::shared_ptr<sclx::detail::page_interface<PageSize>>;
    using page_interface = page_interface<PageSize>;
    using allocation     = allocation<PageSize>;
};

template<page_handle_type HandleType, page_size_t PageSize>
struct page_handle_creator_struct;

template<page_size_t PageSize>
class page_handle<page_handle_type::weak, PageSize> {
  public:
    static constexpr auto handle_type = page_handle_type::weak;
    using alloc_pointer =
        typename page_handle_traits<handle_type, PageSize>::alloc_pointer;
    using page_pointer =
        typename page_handle_traits<handle_type, PageSize>::page_pointer;
    using page_interface =
        typename page_handle_traits<handle_type, PageSize>::page_interface;
    using allocation =
        typename page_handle_traits<handle_type, PageSize>::allocation;

    page_handle() = default;

    page_handle(const page_handle&)            = default;
    page_handle(page_handle&&)                 = default;
    page_handle& operator=(const page_handle&) = default;
    page_handle& operator=(page_handle&&)      = default;

    page_handle(page_handle<page_handle_type::strong, PageSize> other)
        : impl_(std::move(other.impl_)),
          alloc_(std::move(other.alloc_)) {}

    void release() {
        impl_.reset();
        alloc_.reset();
    }

    page_handle<page_handle_type::strong, PageSize> lock() const {
        return page_handle<page_handle_type::strong, PageSize>{
            impl_,
            alloc_.lock()
        };
    }

    ~page_handle() = default;

    friend class page_handle<page_handle_type::weak, PageSize>;
    friend class page_handle<page_handle_type::strong, PageSize>;

    friend struct page_handle_creator_struct<handle_type, PageSize>;

  private:
    page_handle(page_pointer impl, alloc_pointer alloc)
        : impl_(std::move(impl)),
          alloc_(std::move(alloc)) {}

    page_pointer impl_;

    // this pointer keeps any allocation that still has a strong page in
    // use alive, allowing automatic deallocation when the last strong page
    // associated with the allocation is destroyed
    alloc_pointer alloc_;
};

template<page_size_t PageSize>
class page_handle<page_handle_type::strong, PageSize> {
  public:
    static constexpr auto handle_type = page_handle_type::strong;
    using alloc_pointer =
        typename page_handle_traits<handle_type, PageSize>::alloc_pointer;
    using page_pointer =
        typename page_handle_traits<handle_type, PageSize>::page_pointer;
    using page_interface =
        typename page_handle_traits<handle_type, PageSize>::page_interface;
    using allocation =
        typename page_handle_traits<handle_type, PageSize>::allocation;

    page_handle() = default;

    page_handle(const page_handle&)            = default;
    page_handle(page_handle&&)                 = default;
    page_handle& operator=(const page_handle&) = default;
    page_handle& operator=(page_handle&&)      = default;

    std::variant<sclx::byte*, std::future<sclx::byte*>> data() const {
        if (!is_valid()) {
            return nullptr;
        }
        return impl_->data();
    }

    std::variant<device_id_t, sclx::mpi_device> device_id() const {
        if (!is_valid()) {
            return no_device;
        }
        return impl_->device_id();
    }

    std::variant<write_bit_t, std::future<write_bit_t>> write_bit() const {
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

    page_index_t index() const {
        if (!is_valid()) {
            return invalid_page;
        }
        return impl_->index();
    }

    bool is_valid() const { return impl_ != nullptr && alloc_ != nullptr; }

    void release() {
        impl_.reset();
        alloc_.reset();
    }

    sclx::event
    copy_from(sycl::queue q, page_handle src, page_copy_rules rules = {}) {
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
            return q.submit([=](sycl::handler& cgh) {
                cgh.depends_on(skip_copy_event);
                cgh.host_task([&, src]() {
                    if (*skip_copy_ptr) {
                        return;
                    }
                    this->impl_->copy_from(q, src.impl_).wait_and_throw();
                });
            });
        }

        return impl_->copy_from(q, src.impl_);
    }

    bool is_mpi_local() const {
        if (!is_valid()) {
            return true;
        }
        return impl_->is_mpi_local();
    }

    sclx::event replace_with(sycl::queue q, page_handle new_page) {
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
    friend class page_handle<page_handle_type::strong, PageSize>;

    friend struct page_handle_creator_struct<handle_type, PageSize>;

  private:
    page_handle(page_pointer impl, alloc_pointer alloc)
        : impl_(std::move(impl)),
          alloc_(std::move(alloc)) {}

    static std::variant<bool, std::future<bool>> should_skip_copy(
        const page_handle& dst,
        const page_handle& src,
        page_copy_rules rules
    ) {
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

    // this pointer keeps any allocation that still has a strong page in
    // use alive, allowing automatic deallocation when the last strong page
    // associated with the allocation is destroyed
    alloc_pointer alloc_;
};

template<page_handle_type HandleType, page_size_t PageSize>
struct page_handle_creator_struct {
    static page_handle<HandleType, PageSize> create(
        typename page_handle_traits<HandleType, PageSize>::page_pointer impl,
        typename page_handle_traits<HandleType, PageSize>::alloc_pointer alloc
    ) {
        return page_handle<HandleType, PageSize>(impl, alloc);
    }
};

template<page_handle_type HandleType, page_size_t PageSize>
page_handle<HandleType, PageSize> make_page_handle(
    typename page_handle_traits<HandleType, PageSize>::page_pointer impl,
    typename page_handle_traits<HandleType, PageSize>::alloc_pointer alloc
) {
    return page_handle_creator_struct<HandleType, PageSize>::create(
        impl,
        alloc
    );
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
constexpr page_count_t required_pages_for_elements(page_count_t elements) {
    return (elements + page_traits<T, PageSize>::elements_per_page - 1)
         / page_traits<T, PageSize>::elements_per_page;
}

template<
    class T,
    class PageBeginIterator,
    class PageEndIterator,
    class PageDstIterator,
    page_size_t PageSize = default_page_size>
sclx::event copy_pages(
    const sycl::queue& q,
    PageBeginIterator begin,
    PageEndIterator end,
    PageDstIterator dst,
    page_copy_rules rules = {}
) {
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
