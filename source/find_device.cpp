#include <scalix/find_device.hpp>

namespace sclx {

device_id_t find_device(const sclx::device& device) {

    auto device_list = sclx::device::get_devices();
    for (auto& d : device_list) {
        if (d == device) {
            return static_cast<int>(std::distance(&device_list[0], &d) + 1);
        }
    }

    return no_device;
}

}