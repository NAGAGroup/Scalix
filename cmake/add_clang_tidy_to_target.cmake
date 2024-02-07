cmake_minimum_required(VERSION 3.14)

set(SCALIX_USING_CUSTOM_CLANG_TIDY "")

if(NOT DEFINED "CLANG_TIDY_CHECK_SUBTARGETS")
  set(CLANG_TIDY_CHECK_SUBTARGETS
      STRING
      "List of custom clang-tidy-check targets that should be added to the global clang-tidy-check target"
  )
endif()

if(NOT DEFINED "CLANG_TIDY_FIX_SUBTARGETS")
  set(CLANG_TIDY_FIX_SUBTARGETS
      STRING
      "List of custom clang-tidy-fix targets that should be added to the global clang-tidy-fix target"
  )
endif()

function(scalix_add_clang_tidy_to_library target)
  get_target_property(TARGET_SOURCES ${target} SOURCES)
  add_library(${target}-dont-compile ${TARGET_SOURCES})
  get_target_property(TARGET_INCLUDE_DIRS ${target} INCLUDE_DIRECTORIES)
  target_include_directories(${target}-dont-compile
                             PRIVATE ${TARGET_INCLUDE_DIRS})

  if(DEFINED "SCALIX_BASE_CXX_FLAGS")
    target_link_options(${target}-dont-compile BEFORE PUBLIC
                        ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_BASE_CXX_FLAGS")
    target_compile_options(${target}-dont-compile BEFORE
                           PUBLIC ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_TIDY_INCLUDE_ARGS")
    target_link_options(${target}-dont-compile BEFORE PUBLIC
                        ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_TIDY_INCLUDE_ARGS")
    target_compile_options(${target}-dont-compile BEFORE
                           PUBLIC ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_BASE_SYCL_FLAGS")
    target_link_options(${target}-dont-compile BEFORE PUBLIC
                        ${SCALIX_BASE_SYCL_FLAGS})
  endif()

  if(DEFINED "SCALIX_BASE_SYCL_FLAGS")
    target_compile_options(${target}-dont-compile BEFORE
                           PUBLIC ${SCALIX_BASE_SYCL_FLAGS})
  endif()

  include(GenerateExportHeader)
  generate_export_header(
    ${target}-dont-compile
    BASE_NAME
    scalix
    EXPORT_FILE_NAME
    export/${target}-dont-compile/${target}/${target}_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE
    pragma_suppress_c4251)

  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${target}-dont-compile
                               PUBLIC SCALIX_STATIC_DEFINE)
  endif()

  set_target_properties(
    ${target}-dont-compile
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES
               VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               EXCLUDE_FROM_ALL YES
               EXPORT_COMPILE_COMMANDS ON)

  target_include_directories(
    ${target}-dont-compile ${warning_guard}
    PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

  target_include_directories(
    ${target}-dont-compile SYSTEM
    PUBLIC
      "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export/${target}-dont-compile>")

  add_custom_target(
    ${target}-clang-tidy-checks
    ${SCALIX_CXX_CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${TARGET_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_target_properties(${target}-clang-tidy-checks PROPERTIES EXCLUDE_FROM_ALL
                                                               YES)
  add_custom_target(
    ${target}-clang-tidy-fixes
    ${SCALIX_CXX_CLANG_TIDY} -p ${CMAKE_BINARY_DIR} -fix ${TARGET_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_target_properties(${target}-clang-tidy-fixes PROPERTIES EXCLUDE_FROM_ALL
                                                              YES)

  # add targets to the global clang-tidy-checks and clang-tidy-fixes targets
  set(CLANG_TIDY_CHECK_SUBTARGETS
      ${CLANG_TIDY_CHECK_SUBTARGETS} ${target}-clang-tidy-checks
      CACHE
        STRING
        "List of custom clang-tidy-check targets that should be added to the global clang-tidy-check target"
        FORCE)
  set(CLANG_TIDY_FIX_SUBTARGETS
      ${CLANG_TIDY_FIX_SUBTARGETS} ${target}-clang-tidy-fixes
      CACHE
        STRING
        "List of custom clang-tidy-fix targets that should be added to the global clang-tidy-fix target"
        FORCE)
endfunction()

function(scalix_add_clang_tidy_to_executable target)
  get_target_property(TARGET_SOURCES ${target} SOURCES)
  add_executable(${target}-dont-compile ${TARGET_SOURCES})
  get_target_property(TARGET_INCLUDE_DIRS ${target} INCLUDE_DIRECTORIES)
  target_include_directories(${target}-dont-compile
                             PRIVATE ${TARGET_INCLUDE_DIRS})

  if(DEFINED "SCALIX_BASE_CXX_FLAGS")
    target_link_options(${target}-dont-compile BEFORE PUBLIC
                        ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_BASE_CXX_FLAGS")
    target_compile_options(${target}-dont-compile BEFORE
                           PUBLIC ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_TIDY_INCLUDE_ARGS")
    target_link_options(${target}-dont-compile BEFORE PUBLIC
                        ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_TIDY_INCLUDE_ARGS")
    target_compile_options(${target}-dont-compile BEFORE
                           PUBLIC ${SCALIX_BASE_CXX_FLAGS})
  endif()

  if(DEFINED "SCALIX_BASE_SYCL_FLAGS")
    target_link_options(${target}-dont-compile BEFORE PUBLIC
                        ${SCALIX_BASE_SYCL_FLAGS})
  endif()

  if(DEFINED "SCALIX_BASE_SYCL_FLAGS")
    target_compile_options(${target}-dont-compile BEFORE
                           PUBLIC ${SCALIX_BASE_SYCL_FLAGS})
  endif()

  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${target}-dont-compile
                               PUBLIC SCALIX_STATIC_DEFINE)
  endif()

  set_target_properties(
    ${target}-dont-compile
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES
               VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               EXCLUDE_FROM_ALL YES
               EXPORT_COMPILE_COMMANDS ON)

  add_custom_target(
    ${target}-clang-tidy-checks
    ${SCALIX_CXX_CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${TARGET_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_target_properties(${target}-clang-tidy-checks PROPERTIES EXCLUDE_FROM_ALL
                                                               YES)
  add_custom_target(
    ${target}-clang-tidy-fixes
    ${SCALIX_CXX_CLANG_TIDY} -p ${CMAKE_BINARY_DIR} -fix ${TARGET_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_target_properties(${target}-clang-tidy-fixes PROPERTIES EXCLUDE_FROM_ALL
                                                              YES)

  # add targets to the global clang-tidy-checks and clang-tidy-fixes targets
  set(CLANG_TIDY_CHECK_SUBTARGETS
      ${CLANG_TIDY_CHECK_SUBTARGETS} ${target}-clang-tidy-checks
      CACHE
        STRING
        "List of custom clang-tidy-check targets that should be added to the global clang-tidy-check target"
        FORCE)
  set(CLANG_TIDY_FIX_SUBTARGETS
      ${CLANG_TIDY_FIX_SUBTARGETS} ${target}-clang-tidy-fixes
      CACHE
        STRING
        "List of custom clang-tidy-fix targets that should be added to the global clang-tidy-fix target"
        FORCE)
endfunction()
