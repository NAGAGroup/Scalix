cmake_minimum_required(VERSION 3.14)

if (SCALIX_DEVELOPER_MODE AND DEFINED "SCALIX_CLANG_TIDY_COMMAND")
    set(SCALIX_CXX_CLANG_TIDY ${SCALIX_CLANG_TIDY_COMMAND}
            ${SCALIX_CLANG_TIDY_ARGS})
    if (NOT DEFINED "AdaptiveCpp_DIR")
        include(${CMAKE_SOURCE_DIR}/cmake/add_clang_tidy_to_target.cmake)
    else ()
        function(scalix_add_clang_tidy_to_library target)
            return()
        endfunction()

        function(scalix_add_clang_tidy_to_executable target)
            return()
        endfunction()
    endif ()
endif ()

if (NOT DEFINED "SCALIX_EXTRA_SYCL_ARGS")
    set(SCALIX_EXTRA_SYCL_ARGS "")
endif ()

if (DEFINED "AdaptiveCpp_DIR")
    find_package(AdaptiveCpp REQUIRED)
    find_package(Threads REQUIRED)
    set(ACPP_TARGETS "${SCALIX_SYCL_TARGETS}")
else ()
    set(SCALIX_CLANG_SYCL_FLAG "-fsycl")
    set(SCALIX_EXTRA_SYCL_ARGS "-fsycl-targets=${SCALIX_SYCL_TARGETS}"
            ${SCALIX_EXTRA_SYCL_ARGS})
endif ()

function(add_scalix_to_target target)
    # When using Intel's LLVM SYCL compiler, there are some flags that need to be
    # excluded from the flags passed to clang-tidy as they are not known to
    # clang-tidy. This is a workaround, ideally these flags would be known to
    # clang-tidy. The below sets the flags for the true compiled scalix target.
    # Later a subset of these flags are passed to the dummy clang-tidy target. For
    # AdaptiveCpp, this step is unncessesary when using the SSCP mode and the
    # custom clang-tidy target will not be added instead opting to use
    # CMAKE_CXX_CLANG_TIDY directly. So far only SSCP mode is supported, so we
    # will need to explore if this is necessary for other modes.
    target_compile_options(${target} BEFORE PUBLIC ${SCALIX_EXTRA_SYCL_ARGS})
    target_link_options(${target} BEFORE PUBLIC ${SCALIX_EXTRA_SYCL_ARGS})
    if (DEFINED "SCALIX_CLANG_SYCL_FLAG")
        target_compile_options(${target} BEFORE PUBLIC ${SCALIX_CLANG_SYCL_FLAG})
        target_link_options(${target} BEFORE PUBLIC ${SCALIX_CLANG_SYCL_FLAG})
    endif ()

    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(${target} PUBLIC SCALIX_STATIC_DEFINE)
    endif()

    target_include_directories(
            ${target} ${warning_guard}
            PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

    target_include_directories(
            ${target} SYSTEM PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>")

    get_target_property(target_type ${target} TYPE)
    if ("${target_type}" MATCHES "LIBRARY")
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

        if (DEFINED "SCALIX_CXX_CLANG_TIDY")
            scalix_add_clang_tidy_to_library(${target} ${SCALIX_CXX_CLANG_TIDY})
        endif ()
    else ()
        if (DEFINED "SCALIX_CXX_CLANG_TIDY")
            scalix_add_clang_tidy_to_executable(${target} ${SCALIX_CXX_CLANG_TIDY})
        endif ()
    endif ()

    if(DEFINED "AdaptiveCpp_DIR")
        target_compile_definitions(${target} PUBLIC SCALIX_ADAPTIVECPP)
        get_target_property(SCALIX_SOURCES ${target} SOURCES)
        add_sycl_to_target(TARGET ${target} SOURCES ${SCALIX_SOURCES})
        target_link_libraries(${target} PUBLIC Threads::Threads)
    endif()
endfunction()
