cmake_minimum_required(VERSION 3.14)

project(ScalixTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)
include(../cmake/add_scalix_to_target.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(Scalix REQUIRED)
  enable_testing()
endif()

find_package(Catch2 3 REQUIRED)
include(Catch)

# ---- Tests ----

add_executable(tests source/scratchwork.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain cpptrace::cpptrace)
catch_discover_tests(tests)
add_scalix_to_target(tests)

# ---- End-of-file commands ----

add_folders(Test)
