#!/bin/bash

set -eo pipefail

ARGS=(
  -DWITH_INTGEMM=OFF
  -DWITH_RUY=OFF
  -DWITH_GEMMOLOGY=ON
  -DUSE_AVX512=ON -DUSE_AVX2=ON -DUSE_SSSE3=ON -DUSE_SSE2=ON
  -DWITH_BLAS=ON

  # TODO(jerinphilip) Adjust, later.
  -DCMAKE_BUILD_TYPE=Debug
  -DWITH_ASAN=ON

  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

set -x

# Configure
cmake -B build -S $PWD "${ARGS[@]}"

# Build
cmake --build build --target all
