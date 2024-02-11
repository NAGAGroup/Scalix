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

# ---- Tests ----

add_executable(scratchwork source/scratchwork.cpp)
target_link_libraries(scratchwork PRIVATE Scalix::Scalix)
add_scalix_to_target(scratchwork)

add_test(NAME scratchwork COMMAND scratchwork)

# ---- End-of-file commands ----

add_folders(Test)
