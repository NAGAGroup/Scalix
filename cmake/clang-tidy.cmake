cmake_minimum_required(VERSION 3.14)

if(DEFINED "CMAKE_CXX_CLANG_TIDY")
  set(CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};-p;${CMAKE_BINARY_DIR}")
  if(DEFINED "CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR")
    set(CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR "${CMAKE_BINARY_DIR}/${CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR}")
    add_custom_target(
      clang-tidy-fix
      COMMAND cmake --build ${CMAKE_BINARY_DIR}
      COMMAND clang-apply-replacements ${CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR}
              --ignore-insert-conflict
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Build all targets if necessary and apply clang-tidy fixes"
      VERBATIM USES_TERMINAL)
  endif()
endif()
