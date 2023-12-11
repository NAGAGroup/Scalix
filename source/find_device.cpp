#include <iterator>
#include <scalix/defines.hpp>
#include <scalix/find_device.hpp>
#include <scalix/scalix_export.hpp>

namespace sclx {

SCALIX_EXPORT auto find_device(const sclx::device& device) -> device_id_t {

    auto device_list = sclx::device::get_devices();
    for (auto& dev : device_list) {
        if (dev == device) {
            return static_cast<device_id_t>(
                std::distance(device_list.data(), &dev)
            );
        }
    }

    return no_device;
}

}  // namespace sclx