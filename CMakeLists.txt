cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
  Scalix
  VERSION 1.0.0
  DESCRIPTION
    "SYCL-like library for easily distributing data parallel applications across available compute resources"
  HOMEPAGE_URL "https://github.com/NAGAGroup/Scalix"
  LANGUAGES C CXX)

# ---- Library Dependencies ----
include(cmake/dependencies.cmake)

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

include(cmake/configure_target_for_scalix.cmake)

# ---- Clang tidy ----
include(cmake/clang-tidy.cmake)

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
  include/scalix/detail/generic_task.hpp
  source/generic_task.cpp
  include/scalix/generic_task.hpp
  include/scalix/typed_task_interface.hpp
  include/scalix/typed_task.hpp
  source/typed_task.cpp
  include/scalix/concurrent_guard.hpp
  source/concurrent_guard.cpp
  source/detail/page_data.cpp
  include/scalix/detail/page_data.hpp
  source/accessor.cpp
  include/scalix/accessor.hpp)
add_library(Scalix::scalix ALIAS scalix)

# ---- This handles SYCL stuffs ----
configure_target_for_scalix(scalix)

# ---- Configure library ----

# This ensures clang-tidy runs on all source files, including headers
if(NOT BUILD_SHARED_LIBS)
  get_target_property(TARGET_SOURCES scalix SOURCES)
  get_target_property(TARGET_COMPILE_FLAGS scalix COMPILE_FLAGS)
  set_source_files_properties(${TARGET_SOURCES} PROPERTIES LANGUAGE CXX)
  foreach(source_file ${TARGET_SOURCES})
    if(source_file MATCHES ".*\\.hpp")
      set_source_files_properties(${source_file} PROPERTIES HEADER_FILE_ONLY
                                                            FALSE)
    endif()
  endforeach()
else()
  # Developer warning that indicates full clang-tidy checks are not run when
  # building shared libraries
  message(
    AUTHOR_WARNING
      "Clang-tidy checks are not run on header files when building shared libraries. "
      "Header files included in source files are still checked, but clang-tidy doesn't "
      "run all checks on includes that it does on source files. This is really only a problem for templated code."
  )
endif()

if(DEFINED "CUDAToolkit_FOUND")
  target_include_directories(scalix SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
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

# get compiler args for scalix TODO: Manually exporting symbols has proven to be
# a pain, and since this is just early development, we just export all symbols
# for now. It would be nice to have a more fine-grained control over this in the
# future.
set_target_properties(
  scalix
  PROPERTIES # CXX_VISIBILITY_PRESET hidden
             # VISIBILITY_INLINES_HIDDEN YES
             VERSION "${PROJECT_VERSION}" SOVERSION "${PROJECT_VERSION_MAJOR}")

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(scalix PUBLIC SCALIX_STATIC_DEFINE)
endif()

target_include_directories(
  scalix ${warning_guard} SYSTEM
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<INSTALL_INTERFACE:include/Scalix-${PROJECT_VERSION}>")

target_include_directories(
  scalix SYSTEM PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>")

# ---- Dependencies ----
target_link_libraries(scalix PUBLIC cpptrace::cpptrace)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()
