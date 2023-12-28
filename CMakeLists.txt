cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
  Scalix
  VERSION 1.0.0
  DESCRIPTION
    "SYCL-like library for easily distributing data parallel applications across available compute resources"
  HOMEPAGE_URL "https://github.com/NAGAGroup/Scalix"
  LANGUAGES CXX)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Declare library ----

add_library(
  scalix
  source/detail/page_handle.cpp
  source/detail/allocation.cpp
  source/detail/page_table.cpp
  source/detail/device_allocation.cpp
  source/find_device.cpp
  source/detail/device_page.cpp
  source/detail/device_page_table.cpp
  source/detail/host_page_table.cpp
  source/detail/local_page.cpp
  source/detail/host_page.cpp
  source/detail/host_allocation.cpp)
add_library(Scalix::Scalix ALIAS scalix)

if(DEFINED "SCALIX_CXX_FLAGS")
  target_link_options(scalix BEFORE PUBLIC ${SCALIX_CXX_FLAGS})
endif()

if(DEFINED "SCALIX_CXX_FLAGS")
  target_compile_options(scalix BEFORE PUBLIC ${SCALIX_CXX_FLAGS})
endif()

include(GenerateExportHeader)
generate_export_header(
  scalix
  BASE_NAME
  scalix
  EXPORT_FILE_NAME
  export/scalix/scalix_export.hpp
  CUSTOM_CONTENT_FROM_VARIABLE
  pragma_suppress_c4251)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(scalix PUBLIC SCALIX_STATIC_DEFINE)
endif()

# get compiler args for scalix
set_target_properties(
  scalix
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES
             VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             EXPORT_NAME Scalix
             OUTPUT_NAME Scalix)

target_include_directories(
  scalix ${warning_guard}
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

target_include_directories(
  scalix SYSTEM PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>")

target_compile_features(scalix PUBLIC cxx_std_20)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT Scalix_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(AUTHOR_WARNING "Developer mode is intended for developers of Scalix")
endif()

include(cmake/dev-mode.cmake)
