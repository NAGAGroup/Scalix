#!/bin/bash
find . -regex './scalix/.*\.\(cuh\|cu\|hpp\|cpp\|h\|c\)' -exec clang-format -style=file -i {} \;
find . -regex './examples/.*\.\(cuh\|cu\|hpp\|cpp\|h\|c\)' -exec clang-format -style=file -i {} \;
cmake-format CMakeLists.txt -o CMakeLists.txt