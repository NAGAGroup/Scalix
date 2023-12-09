cmake_minimum_required(VERSION 3.14)

project(ScalixTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(Scalix REQUIRED)
  enable_testing()
endif()

# ---- Tests ----

add_executable(scratchwork source/scratchwork.cpp)
target_link_libraries(scratchwork PRIVATE Scalix::Scalix)
target_compile_features(scratchwork PRIVATE cxx_std_20)
if(DEFINED "SCALIX_SYCL_CXX_FLAGS")
  target_compile_options(scratchwork BEFORE PRIVATE ${SCALIX_SYCL_CXX_FLAGS})
endif()

add_test(NAME scratchwork COMMAND scratchwork)

# ---- End-of-file commands ----

add_folders(Test)
