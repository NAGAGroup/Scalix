cmake_minimum_required(VERSION 3.20)
project(
  SCALIX
  VERSION 0.1.0
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# require CUDA 12.0 or higher
find_package(CUDAToolkit 12.0 REQUIRED)

# scalix is a header-only library
add_library(scalix INTERFACE)
target_include_directories(scalix INTERFACE ${PROJECT_SOURCE_DIR}/.)
target_compile_options(scalix INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
                                        --extended-lambda --expt-relaxed-constexpr >)

# if SCALIX_DEBUG_KERNEL_LAUNCH is on, then useful diagnostic information is
# printed to stdout when a kernel is launched.
option(SCALIX_DEBUG_KERNEL_LAUNCH "Enable debug output for kernel launches" OFF)
if(SCALIX_DEBUG_KERNEL_LAUNCH)
  target_compile_definitions(scalix INTERFACE SCALIX_DEBUG_KERNEL_LAUNCH)
endif()

# if SCALIX_EMULATE_MULTIDEVICE is set to a value other than 0, then
# SCALIX_EMULATE_MULTIDEVICE is defined to that value which will emulate the
# provided number of devices.  This is useful for testing on a single device
# system.
if(DEFINED SCALIX_EMULATE_MULTIDEVICE)
  if(NOT (SCALIX_EMULATE_MULTIDEVICE EQUAL 0))
    message(
      STATUS
        "SCALIX_EMULATE_MULTIDEVICE is set to ${SCALIX_EMULATE_MULTIDEVICE}")
    target_compile_definitions(
      scalix INTERFACE SCALIX_EMULATE_MULTIDEVICE=${SCALIX_EMULATE_MULTIDEVICE})
  endif()
endif()

# examples
add_executable(distributed_indexing examples/distributed_indexing.cu)
target_link_libraries(distributed_indexing scalix)

add_executable(inclusive_scan examples/inclusive_scan.cu)
target_link_libraries(inclusive_scan scalix)

add_executable(index_generator examples/index_generator.cu)
target_link_libraries(index_generator scalix)

add_executable(reduce_last_dim examples/reduce_last_dim.cu)
target_link_libraries(reduce_last_dim scalix)

add_executable(scaling_performance examples/scaling_performance.cu)
target_link_libraries(scaling_performance scalix)
