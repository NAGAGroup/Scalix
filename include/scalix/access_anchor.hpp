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
#include "typed_task.hpp"

#include <memory>
#include <scalix/accessor.hpp>
#include <scalix/defines.hpp>
#include <scalix/detail/page_data.hpp>
#include <scalix/generic_task.hpp>
#include <scalix/pointers.hpp>
#include <unordered_map>

namespace sclx {

class object_id {
public:
    auto operator==(const object_id&) const -> bool = default;
    auto operator!=(const object_id&) const -> bool = default;

private:
    void* id_{nullptr};
};

class access_anchor;
class distributed_handler;

template<class T, int Dimensions>
class distributed_buffer;

template<class T, int Dimensions>
class access_strategy_base;

template<class AccessStrategy, class T, int Dimensions>
concept AccessStrategyConcept = requires(AccessStrategy strategy) {
    {
        strategy
            .template get<T, Dimensions>(
                std::declval<const std::vector<device>&>(),
                std::declval<distributed_buffer<T, Dimensions>>()
            )
            ->std::template shared_ptr<access_strategy_base<T, Dimensions>>
    };
};

class access_anchor {
  public:
    template<class, int>
    friend class distributed_buffer;

    friend class distributed_handler;

    access_anchor() = default;

    access_anchor(const access_anchor&)            = default;
    access_anchor& operator=(const access_anchor&) = default;

    access_anchor(access_anchor&&)            = default;
    access_anchor& operator=(access_anchor&&) = default;

    ~access_anchor() = default;

  private:
    struct device_anchor {
        // the page anchors are type erased to avoid any PageSize template
        // arguments for the access_anchor, but note that the underlying
        // datatype is page_data_interface<PageSize>, where this conversion
        // is done from within a class that knows the PageSize
        std::vector<std::shared_ptr<void>> page_anchors_;
        shared_ptr<page_ptr_t[]> device_page_ptrs_{nullptr
        };  // NOLINT(*-avoid-c-arrays)
    };

    std::shared_ptr<std::unordered_map<device, device_anchor>> device_anchors_
        = std::make_shared<std::unordered_map<device, device_anchor>>();

    void set_device_page_anchors(
        const device& dev,
        std::vector<std::shared_ptr<void>> page_anchors
    ) const {
        device_anchors_->at(dev).page_anchors_ = std::move(page_anchors);
    }

    [[nodiscard]] device_anchor& get_device_anchor(const device& dev) const {
        return device_anchors_->at(dev);
    }

    void combine_with(const access_anchor& other) {
        if (other.device_anchors_ == nullptr) {
            return;
        }

        if (device_anchors_ == nullptr) {
            device_anchors_ = other.device_anchors_;
            return;
        }

        for (const auto& [dev, anchor] : *other.device_anchors_) {
            if (!device_anchors_->contains(dev)) {
                device_anchors_->emplace(dev, anchor);
            } else {
                throw std::runtime_error(
                    "device already exists in anchor, indicates bug in Scalix"
                );
            }
        }
    }
};

class distributed_handler;

class access_strategy_queue_interface {
  public:
    friend class distributed_handler;

    template<class, int>
    friend class distributed_buffer;

    access_strategy_queue_interface() = default;

    access_strategy_queue_interface(const access_strategy_queue_interface&)
        = default;
    access_strategy_queue_interface&
    operator=(const access_strategy_queue_interface&)
        = default;

    access_strategy_queue_interface(access_strategy_queue_interface&&)
        = default;
    access_strategy_queue_interface&
    operator=(access_strategy_queue_interface&&)
        = default;

    virtual ~access_strategy_queue_interface() = default;

  private:
    [[nodiscard]] virtual auto get_anchor() const -> const access_anchor& = 0;

    [[nodiscard]] virtual auto configure_access_anchor(
        const device& dev,
        const id<>& device_range_offset,
        const range<>& device_range
    ) const -> generic_task
        = 0;

    [[nodiscard]] virtual auto configure_access_anchor(
        const device& dev,
        const id<2>& device_range_offset,
        const range<2>& device_range
    ) const -> generic_task
        = 0;

    [[nodiscard]] virtual auto configure_access_anchor(
        const device& dev,
        const id<3>& device_range_offset,
        const range<3>& device_range
    ) const -> generic_task
        = 0;

    virtual auto post_command_task(const device& dev) -> generic_task = 0;
};

template<class T, int Dimensions>
class access_strategy_base : public access_strategy_queue_interface {
  public:
    using access_strategy_queue_interface::access_strategy_queue_interface;

  protected:
    distributed_buffer<T, Dimensions> distributed_buffer_;
    access_anchor anchor_;

    access_strategy_base(
        distributed_buffer<T, Dimensions> distributed_buffer,
        access_anchor anchor
    )
        : distributed_buffer_(std::move(distributed_buffer)),
          anchor_(std::move(anchor)) {}

  private:
    [[nodiscard]] auto get_anchor() const -> const access_anchor& final {
        return anchor_;
    }

    [[nodiscard]] auto post_command_task(const device& dev)
        -> generic_task final {
        return distributed_buffer_.post_command_task(dev, anchor_);
    }
};

class default_access_strategy {
    template<class T, int Dimensions>
    friend class distributed_buffer;

    template<class T, int Dimensions>
    class strategy final : access_strategy_base<T, Dimensions> {
      public:
        using access_strategy_base<T, Dimensions>::access_strategy_base;

      private:
        [[nodiscard]] auto configure_access_anchor(
            const device& dev,
            const id<>& device_range_offset,
            const range<>& device_range
        ) const -> generic_task override {
            return configure_access_anchor(dev);
        }

        [[nodiscard]] auto configure_access_anchor(
            const device& dev,
            const id<2>& device_range_offset,
            const range<2>& device_range
        ) const -> generic_task override {
            return configure_access_anchor(dev);
        }

        [[nodiscard]] auto configure_access_anchor(
            const device& dev,
            const id<3>& device_range_offset,
            const range<3>& device_range
        ) const -> generic_task override {
            return configure_access_anchor(dev);
        }

        [[nodiscard]] auto configure_access_anchor(const device& dev) const
            -> generic_task {
            auto task = create_task([=, *this] {
                auto page_count = this->distributed_buffer_.page_count();
                auto page_size  = this->distributed_buffer_.page_size();
                auto util_bytes_per_page
                    = detail::utilized_bytes_per_page<T>(page_size);
                auto num_bytes_to_alloc
                    = static_cast<size_t>(page_count)
                    * static_cast<size_t>(util_bytes_per_page);
                auto alloc = make_shared<byte[]>(
                    sycl::queue{dev},
                    usm::alloc::device,
                    num_bytes_to_alloc
                );
                std::vector<byte*> pages(page_count);
                for (size_t i = 0; i < static_cast<size_t>(page_count); ++i) {
                    pages[i] = alloc.get() + i * util_bytes_per_page;
                }

                this->distributed_buffer_.populate_page_anchors(
                    this->anchor_,
                    alloc,
                    std::move(pages)
                );
            });
            return task;
        }
    };

    template<class T, int Dimensions>
    [[nodiscard]] auto
    get(distributed_buffer<T, Dimensions> distributed_buffer,
        access_anchor anchor)
        -> std::shared_ptr<access_strategy_base<T, Dimensions>> {
        return std::make_shared<strategy<T, Dimensions>>(
            strategy<T, Dimensions>(
                std::move(distributed_buffer),
                std::move(anchor)
            )
        );
    }
};

template<class T, int Dimensions>
class distributed_buffer {
  public:
    friend class access_strategy_base<T, Dimensions>;

    template<
        access_mode AccessMode = access_mode::read_write,
        sycl::target Targ      = sycl::target::device,
        class AccessStrategy   = default_access_strategy>
        requires AccessStrategyConcept<AccessStrategy, T, Dimensions>
    [[nodiscard]] auto get_access(
        distributed_handler& cgh,
        AccessStrategy access_strategy = default_access_strategy{}
    ) const -> accessor<T, Dimensions, AccessMode>;

  private:
    [[nodiscard]] auto
    post_command_task(const device& dev, const access_anchor& anchor)
        -> generic_task;

    [[nodiscard]] auto page_size() const -> page_size_t;

    [[nodiscard]] auto page_count() const -> page_count_t;

    void populate_page_anchors(
        const access_anchor& anchor,
        std::shared_ptr<void> allocation,
        std::vector<page_ptr_t> pages
    ) const;
};

}  // namespace sclx
