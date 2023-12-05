#include <numeric>
#include <scalix/detail/device_allocation.hpp>
#include <scalix/detail/device_page_table.hpp>

int main() {
    using value_type = float;
    auto page_count = sclx::detail::required_pages_for_elements<value_type>(1'000'000'0);
    std::cout << "page_count: " << page_count << std::endl;
    sclx::device device;
    // get a cuda device
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu()) {
            device = sclx::device(d);
            break;
        }
    }

    sclx::detail::device_page_table page_table{device, page_count};

    sclx::detail::allocation_factory<float> factory;
    factory.allocate_pages_and_reuse_if_possible<sclx::detail::continguous_device_allocation>(
        sclx::find_device(device),
        sclx::page_index_t{0},
        static_cast<sclx::page_index_t>(page_count),
        {}
    );

    std::vector<sclx::page_index_t> indices(page_count);
    std::iota(indices.begin(), indices.end(), 0);
    factory.allocate_pages_and_reuse_if_possible<sclx::detail::continguous_device_allocation>(
        sclx::find_device(device),
        indices,
        {}
    );

    auto page_handles = factory.pages();
    std::vector<sclx::event> events(page_handles.size());
    std::transform(
        page_handles.begin(),
        page_handles.end(),
        events.begin(),
        [&](auto& page_handle) { return page_table.map_page(page_handle); }
    );

    for (auto& event : events) {
        event.wait_and_throw();
    }

    return 0;
}