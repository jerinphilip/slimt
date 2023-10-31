#!/bin/bash

set -eo pipefail

# Configure
ARGS=(
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DWITH_INTGEMM=OFF -DWITH_RUY=OFF
  -DWITH_GEMMOLOGY=ON
  -DUSE_AVX2=ON -DUSE_SSSE3=ON -DUSE_SSE2=ON
  -DUSE_BUILTIN_SENTENCEPIECE=OFF

  -DCMAKE_BUILD_TYPE=Debug
  -DWITH_ASAN=ON
)

cmake -B build -S $PWD "${ARGS[@]}"

# Build
cmake --build build --target all
