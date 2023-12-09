cmake_minimum_required(VERSION 3.14)

macro(default name)
  if(NOT DEFINED "${name}")
    set("${name}" "${ARGN}")
  endif()
endmacro()

default(FIX NO)

set(flag --check)
if(FIX)
  set(flag --write)
endif()

execute_process(
  COMMAND "npx" "prettier" "--prose-wrap=always" "--print-width=80" "${flag}"
          "./*.md" "./notes/**/*.md" RESULT_VARIABLE result)

if(NOT FIX AND result EQUAL "1")
  message(
    FATAL_ERROR
      "Some files are badly formatted. Run again with FIX=YES to fix these files."
  )
elseif(NOT result EQUAL "0")
  message(FATAL_ERROR "Prettier returned with ${result}")
endif()
