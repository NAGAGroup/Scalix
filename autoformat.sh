#!/bin/bash
find . -regex './include/.*\.\(cuh\|cu\|hpp\|cpp\|h\|c\|inl\)' -exec clang-format -style=file -i {} \;
find . -regex './source/.*\.\(cuh\|cu\|hpp\|cpp\|h\|c\|inl\)' -exec clang-format -style=file -i {} \;
find . -regex './examples/.*\.\(cuh\|cu\|hpp\|cpp\|h\|c\|inl\)' -exec clang-format -style=file -i {} \;
cmake-format CMakeLists.txt -o CMakeLists.txt
npx prettier --prose-wrap=always --write --print-width=80 ./**/*.md