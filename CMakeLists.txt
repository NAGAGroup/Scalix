cmake_minimum_required(VERSION 3.20)
project(
  SCALIX
  VERSION 1.0.0
  LANGUAGES CXX)

set(CMAKE_CUDA_STANDARD 17)

add_library(
  scalix SHARED
        source/detail/page_handle.cpp
        include/scalix/detail/page_handle.hpp
        include/scalix/detail/allocation.hpp
        source/detail/allocation.cpp
        include/scalix/detail/page_table.hpp
        source/detail/page_table.cpp
        include/scalix/detail/device_allocation.hpp
        source/detail/device_allocation.cpp
        include/scalix/find_device.hpp
        source/find_device.cpp
        include/scalix/detail/device_page.hpp
        source/detail/device_page.cpp
        include/scalix/detail/device_page_table.hpp
        source/detail/device_page_table.cpp)

target_include_directories(scalix PUBLIC include)

add_executable(scratchwork scratchwork.cpp)
target_link_libraries(scratchwork scalix)
