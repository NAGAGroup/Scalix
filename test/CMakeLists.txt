cmake_minimum_required(VERSION 3.14)

project(ScalixTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  enable_testing()
  find_package(Scalix REQUIRED)
else()
  include(../cmake/configure_target_for_scalix.cmake)
endif()

find_package(Catch2 3 REQUIRED)
include(Catch)

# ---- Tests ----

add_executable(tests source/typed_task.cpp
  source/concurrent_guard.cpp
  source/page_data.cpp
)
configure_target_for_scalix(tests)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Scalix::scalix)
catch_discover_tests(tests)

add_executable(exploratory source/exploratory.cpp)
configure_target_for_scalix(exploratory)
target_link_libraries(exploratory PRIVATE Scalix::scalix)

# ---- End-of-file commands ----

add_folders(Test)
