set(CLANG_FORMAT_PATTERNS
    source/*.cpp source/*.hpp include/*.hpp test/*.cpp test/*.hpp
    CACHE STRING
          "; separated patterns relative to the project source dir to format")

set(CLANG_FORMAT_COMMAND
    clang-format
    CACHE STRING "Formatter to use")

add_custom_target(
  clang-format-check
  COMMAND
    "${CMAKE_COMMAND}" -D "CLANG_FORMAT_COMMAND=${CLANG_FORMAT_COMMAND}" -D
    "CLANG_FORMAT_PATTERNS=${CLANG_FORMAT_PATTERNS}" -P
    "${PROJECT_SOURCE_DIR}/cmake/clang-format.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Linting the code"
  VERBATIM)

add_custom_target(
  clang-format-fix
  COMMAND
    "${CMAKE_COMMAND}" -D "CLANG_FORMAT_COMMAND=${CLANG_FORMAT_COMMAND}" -D
    "CLANG_FORMAT_PATTERNS=${CLANG_FORMAT_PATTERNS}" -D FIX=YES -P
    "${PROJECT_SOURCE_DIR}/cmake/clang-format.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Fixing the code"
  VERBATIM)

set(CMAKEFORMAT_PATTERNS
    CMakeLists.txt cmake/*.cmake
    CACHE STRING
          "; separated patterns relative to the project source dir to format")

set(CMAKEFORMAT_COMMAND
    cmake-format
    CACHE STRING "Formatter to use")

add_custom_target(
  cmake-format-check
  COMMAND
    "${CMAKE_COMMAND}" -D "CMAKEFORMAT_COMMAND=${CMAKEFORMAT_COMMAND}" -D
    "CMAKEFORMAT_PATTERNS=${CMAKEFORMAT_PATTERNS}" -P
    "${PROJECT_SOURCE_DIR}/cmake/cmake-format.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Linting cmake files"
  VERBATIM)

add_custom_target(
  cmake-format-fix
  COMMAND
    "${CMAKE_COMMAND}" -D "CMAKEFORMAT_COMMAND=${CMAKEFORMAT_COMMAND}" -D
    "CMAKEFORMAT_PATTERNS=${CMAKEFORMAT_PATTERNS}" -D FIX=YES -P
    "${PROJECT_SOURCE_DIR}/cmake/cmake-format.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Fixing cmake files"
  VERBATIM)

add_custom_target(
  prettier-check
  COMMAND "${CMAKE_COMMAND}" -P "${PROJECT_SOURCE_DIR}/cmake/prettier.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Linting markdown files"
  VERBATIM)

add_custom_target(
  prettier-fix
  COMMAND "${CMAKE_COMMAND}" -D FIX=YES -P
          "${PROJECT_SOURCE_DIR}/cmake/prettier.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Fixing markdown files"
  VERBATIM)

add_custom_target(
  lint-check-all
  DEPENDS clang-format-check cmake-format-check prettier-check
  COMMENT "Checking all files"
  VERBATIM)

add_custom_target(
  lint-fix-all
  DEPENDS clang-format-fix cmake-format-fix prettier-fix
  COMMENT "Fixing all files"
  VERBATIM)
