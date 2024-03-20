include(cmake/folders.cmake)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

option(BUILD_MCSS_DOCS "Build documentation using Doxygen and m.css" OFF)
if(BUILD_MCSS_DOCS)
  include(cmake/docs.cmake)
endif()

option(ENABLE_COVERAGE "Enable coverage support separate from CTest's" OFF)
if(ENABLE_COVERAGE)
  include(cmake/coverage.cmake)
endif()

include(cmake/lint-targets.cmake)
include(cmake/spell-targets.cmake)

# clang-tidy-fix
if(DEFINED "CMAKE_CXX_CLANG_TIDY" AND DEFINED
                                      "CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR")
  add_custom_target(
    clang-tidy-fix
    COMMAND
      cmake --build . && clang-apply-replacements
      ${CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR} --format --style=file
      --style-config=${CMAKE_CURRENT_SOURCE_DIR}/.clang-format
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Build all targets if necessary and apply clang-tidy fixes"
    VERBATIM USES_TERMINAL)
endif()

add_folders(Project)
