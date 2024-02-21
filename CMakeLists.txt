cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
  Scalix
  VERSION 1.0.0
  DESCRIPTION
    "SYCL-like library for easily distributing data parallel applications across available compute resources"
  HOMEPAGE_URL "https://github.com/NAGAGroup/Scalix"
  LANGUAGES CXX)

# check that standard is greater than or equal to 20, if not, raise an error if
# unasigned, set to 20
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
elseif(CMAKE_CXX_STANDARD LESS 20)
  message(
    FATAL_ERROR
      "Scalix requires C++20 or later. You are using C++${CMAKE_CXX_STANDARD}.")
endif()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/add_scalix_to_target.cmake)

# ---- Developer mode ----
if(SCALIX_DEVELOPER_MODE)
  if(NOT PROJECT_IS_TOP_LEVEL)
    message(
      AUTHOR_WARNING "Developer mode is intended for developers of Scalix")
  else()
    message(STATUS "Enabling developer mode for Scalix")
    include(cmake/dev-mode.cmake)
  endif()
endif()

# ---- Declare library ----

add_library(
  scalix
  source/detail/page_handle.cpp
  include/scalix/detail/page_handle.hpp
  source/detail/allocation.cpp
  include/scalix/detail/allocation.hpp
  source/detail/page_table.cpp
  include/scalix/detail/page_table.hpp
  source/detail/device_allocation.cpp
  include/scalix/detail/device_allocation.hpp
  source/find_device.cpp
  include/scalix/find_device.hpp
  source/detail/device_page.cpp
  include/scalix/detail/device_page.hpp
  source/detail/device_page_table.cpp
  include/scalix/detail/device_page_table.hpp
  source/detail/host_page_table.cpp
  include/scalix/detail/host_page_table.hpp
  source/detail/local_page.cpp
  include/scalix/detail/local_page.hpp
  source/detail/host_page.cpp
  include/scalix/detail/host_page.hpp
  source/detail/host_allocation.cpp
  include/scalix/detail/host_allocation.hpp
  source/detail/host_task.cpp
  include/scalix/detail/host_task.hpp
  source/detail/state_machine.cpp
  include/scalix/detail/state_machine.hpp
  source/detail/concurrent_guard.cpp
  include/scalix/detail/concurrent_guard.hpp
  source/detail/state_based_scheduler.cpp
  include/scalix/detail/state_based_scheduler.hpp)
add_library(Scalix::Scalix ALIAS scalix)
add_scalix_to_target(scalix)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Custom Clang Tidy if Needed ----
if(DEFINED "SCALIX_USING_CUSTOM_CLANG_TIDY")
  add_custom_target(
    clang-tidy-checks
    ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
    ${CLANG_TIDY_CHECK_SUBTARGETS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_custom_target(
    clang-tidy-fixes
    ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
    ${CLANG_TIDY_FIX_SUBTARGETS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
