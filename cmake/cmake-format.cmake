cmake_minimum_required(VERSION 3.14)

macro(default name)
  if(NOT DEFINED "${name}")
    set("${name}" "${ARGN}")
  endif()
endmacro()

default(CMAKEFORMAT_COMMAND cmake-format)
default(CMAKEFORMAT_PATTERNS CMakeLists.txt cmake/*.cmake)
default(FIX NO)

set(flag --check)
set(args OUTPUT_VARIABLE output)
if(FIX)
  set(flag -i)
  set(args "")
endif()

file(GLOB files ${CMAKEFORMAT_PATTERNS})
set(badly_formatted "")
set(output "")
string(LENGTH "${CMAKE_SOURCE_DIR}/" path_prefix_length)

foreach(file IN LISTS files)
  if(NOT FIX)
    execute_process(
      COMMAND "${CMAKEFORMAT_COMMAND}" "${file}" "${flag}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      RESULT_VARIABLE result ${args})
    if(result EQUAL "1")
      string(SUBSTRING "${file}" "${path_prefix_length}" -1 relative_file)
      list(APPEND badly_formatted "${relative_file}")
    elseif(NOT result EQUAL "0")
      message(FATAL_ERROR "'${file}': formatter returned with ${result}")
    endif()
    set(output "")
  else()
    execute_process(
      COMMAND "${CMAKEFORMAT_COMMAND}" "${file}" "${flag}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      RESULT_VARIABLE result ${args})
    if(NOT result EQUAL "0")
      message(FATAL_ERROR "'${file}': formatter returned with ${result}")
    endif()
    set(output "")
  endif()
endforeach()

if(NOT badly_formatted STREQUAL "")
  list(JOIN badly_formatted "\n" bad_list)
  message("The following files are badly formatted:\n\n${bad_list}\n")
  message(FATAL_ERROR "Run again with FIX=YES to fix these files.")
endif()
