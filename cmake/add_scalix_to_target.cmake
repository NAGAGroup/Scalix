cmake_minimum_required(VERSION 3.14)

if(SCALIX_DEVELOPER_MODE AND DEFINED "SCALIX_CLANG_TIDY_COMMAND")
  set(SCALIX_CXX_CLANG_TIDY ${SCALIX_CLANG_TIDY_COMMAND}
                            ${SCALIX_CLANG_TIDY_ARGS})
  if(NOT DEFINED "AdaptiveCpp_DIR")
    include(${CMAKE_SOURCE_DIR}/cmake/add_clang_tidy_to_target.cmake)
  else()
    function(scalix_add_clang_tidy_to_library target)
      return()
    endfunction()

    function(scalix_add_clang_tidy_to_executable target)
      return()
    endfunction()
  endif()
endif()

if(NOT DEFINED "SCALIX_EXTRA_SYCL_ARGS")
  set(SCALIX_EXTRA_SYCL_ARGS "")
endif()

if(DEFINED "AdaptiveCpp_DIR")
  find_package(AdaptiveCpp REQUIRED)
  find_package(Threads REQUIRED)
  if(NOT DEFINED "SCALIX_SYCL_TARGETS")
    message(WARNING "No SYCL targets defined, using default generic SSCP mode")
    set(SCALIX_SYCL_TARGETS "generic")
  endif()
  set(ACPP_TARGETS "${SCALIX_SYCL_TARGETS}")
else()
  if(NOT DEFINED "SCALIX_SYCL_TARGETS")
    message(WARNING "No SYCL targets defined, using default spir64_x86_64")
    set(SCALIX_SYCL_TARGETS "spir64_x86_64")
  endif()
  if(NOT DEFINED "SCALIX_EXTRA_SYCL_ARGS")
    set(SCALIX_EXTRA_SYCL_ARGS "")
  endif()
  set(SCALIX_CXX_FLAGS "-fsycl" "-fsycl-targets=${SCALIX_SYCL_TARGETS}"
                       "${SCALIX_EXTRA_SYCL_ARGS}")
endif()

function(add_scalix_to_target target)
  if(DEFINED "SCALIX_CXX_FLAGS")
    target_compile_options(${target} PUBLIC ${SCALIX_CXX_FLAGS})
    target_link_options(${target} PUBLIC ${SCALIX_CXX_FLAGS})
  endif()

  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${target} PUBLIC SCALIX_STATIC_DEFINE)
  endif()

  target_include_directories(
    ${target} ${warning_guard}
    PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

  target_include_directories(
    ${target} SYSTEM PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>")

  get_target_property(target_type ${target} TYPE)
  if("${target_type}" MATCHES "LIBRARY")
    include(GenerateExportHeader)
    generate_export_header(
      ${target}
      BASE_NAME
      scalix
      EXPORT_FILE_NAME
      export/${target}/${target}_export.hpp
      CUSTOM_CONTENT_FROM_VARIABLE
      pragma_suppress_c4251)

    # get compiler args for scalix
    set_target_properties(
      scalix
      PROPERTIES CXX_VISIBILITY_PRESET hidden
                 VISIBILITY_INLINES_HIDDEN YES
                 VERSION "${PROJECT_VERSION}"
                 SOVERSION "${PROJECT_VERSION_MAJOR}")
  endif()

  if(DEFINED "AdaptiveCpp_DIR")
    target_compile_definitions(${target} PUBLIC SCALIX_ADAPTIVECPP)
    get_target_property(SCALIX_SOURCES ${target} SOURCES)
    add_sycl_to_target(TARGET ${target} SOURCES ${SCALIX_SOURCES})
    target_link_libraries(${target} PUBLIC Threads::Threads)
  endif()
endfunction()
