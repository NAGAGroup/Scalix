cmake_minimum_required(VERSION 3.14)
include(CMakeFindDependencyMacro)

include(FetchContent)

set(SCALIX_FETCHED_DEPS "")

find_package(cpptrace 0.5.0 QUIET)
set(SCALIX_FETCHED_CPPTRACE FALSE)
if(NOT cpptrace_FOUND)
  FetchContent_Declare(
    cpptrace
    GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG v0.5.0 # <HASH or TAG>
  )
  list(APPEND SCALIX_FETCHED_DEPS cpptrace)
  set(SCALIX_FETCHED_CPPTRACE TRUE)
endif()

if (SCALIX_FETCHED_DEPS)
  FetchContent_MakeAvailable(${SCALIX_FETCHED_DEPS})
endif()

if (SCALIX_FETCHED_CPPTRACE)
  # copy include directory to build directory
  file(COPY ${cpptrace_SOURCE_DIR}/include/ DESTINATION ${cpptrace_BINARY_DIR}/include/)
endif()

