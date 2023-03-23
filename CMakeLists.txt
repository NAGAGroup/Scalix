cmake_minimum_required(VERSION 3.20)
project(
  SCALIX
  VERSION 0.1.0
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# require CUDA 12.0 or higher
find_package(CUDAToolkit 12.0 REQUIRED)

# if SCALIX_EMULATE_MULTIDEVICE is set to a value other than 0, then
# SCALIX_EMULATE_MULTIDEVICE is defined to that value which will emulate the
# provided number of devices.  This is useful for testing on a single device
# system.
if(DEFINED SCALIX_EMULATE_MULTIDEVICE)
  if(NOT (SCALIX_EMULATE_MULTIDEVICE EQUAL 0))
    message(
      STATUS
        "SCALIX_EMULATE_MULTIDEVICE is set to ${SCALIX_EMULATE_MULTIDEVICE}")
    add_compile_definitions(
      SCALIX_EMULATE_MULTIDEVICE=${SCALIX_EMULATE_MULTIDEVICE})
  endif()
endif()

# if SCALIX_DEBUG_KERNEL_LAUNCH is on, then useful diagnostic information is
# printed to stdout when a kernel is launched.
option(SCALIX_DEBUG_KERNEL_LAUNCH "Enable debug output for kernel launches" OFF)
if(SCALIX_DEBUG_KERNEL_LAUNCH)
  add_compile_definitions(SCALIX_DEBUG_KERNEL_LAUNCH)
endif()

# scalix is a header-only library
add_library(scalix INTERFACE)
target_include_directories(scalix INTERFACE ${PROJECT_SOURCE_DIR}/.)
target_compile_options(scalix INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
                                        --extended-lambda >)

# examples
add_executable(distributed_indexing examples/distributed_indexing.cu)
target_link_libraries(distributed_indexing scalix)
