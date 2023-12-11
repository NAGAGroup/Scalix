cmake_minimum_required(VERSION 3.14)

get_target_property(SCALIX_SOURCES scalix SOURCES)

add_library(scalix-dont-compile ${SCALIX_SOURCES})

target_link_options(scalix-dont-compile BEFORE PRIVATE
                    ${SCALIX_CLANG_TIDY_CXX_FLAGS})

if(DEFINED "SCALIX_TIDY_INCLUDE_ARGS")
  set(SCALIX_CLANG_TIDY_CXX_FLAGS ${SCALIX_TIDY_INCLUDE_ARGS}
                                  ${SCALIX_CLANG_TIDY_CXX_FLAGS})
endif()

target_compile_options(scalix-dont-compile BEFORE
                       PRIVATE ${SCALIX_CLANG_TIDY_CXX_FLAGS})

include(GenerateExportHeader)
generate_export_header(
  scalix-dont-compile
  BASE_NAME
  scalix
  EXPORT_FILE_NAME
  export/scalix-dont-compile/scalix/scalix_export.hpp
  CUSTOM_CONTENT_FROM_VARIABLE
  pragma_suppress_c4251)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(scalix-dont-compile PUBLIC SCALIX_STATIC_DEFINE)
endif()

set_target_properties(
  scalix-dont-compile
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES
             VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             EXPORT_NAME ScalixDontCompile
             OUTPUT_NAME ScalixDontCompile
             EXCLUDE_FROM_ALL YES
             EXPORT_COMPILE_COMMANDS ON)

target_include_directories(
  scalix-dont-compile ${warning_guard}
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

target_include_directories(
  scalix-dont-compile SYSTEM
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export/scalix-dont-compile>")

target_compile_features(scalix-dont-compile PUBLIC cxx_std_20)

add_custom_target(
  clang-tidy-checks
  ${SCALIX_CXX_CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${SCALIX_SOURCES}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(
  clang-tidy-fixes
  ${SCALIX_CXX_CLANG_TIDY} -p ${CMAKE_BINARY_DIR} -fix ${SCALIX_SOURCES}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
