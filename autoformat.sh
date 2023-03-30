#!/bin/bash
find . -regex './scalix/.*\.\(cuh\|cu\|hpp\|cpp\|h\|c\|inl\)' -exec clang-format -style=file -i {} \;
find . -regex './examples/.*\.\(cuh\|cu\|hpp\|cpp\|h\|c\|inl\)' -exec clang-format -style=file -i {} \;
cmake-format CMakeLists.txt -o CMakeLists.txt
prettier --prose-wrap=always --write --print-width=80 ./*.md