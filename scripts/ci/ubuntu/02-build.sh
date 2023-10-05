#!/bin/bash

set -eo pipefail

ARGS=(
  -DWITH_INTGEMM=OFF
  -DWITH_RUY=OFF
  -DWITH_GEMMOLOGY=ON -DUSE_AVX2=ON
  -DWITH_BLAS=ON
)

set -x

# Configure
cmake -B build -S $PWD -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${ARGS[@]}"

# Build
cmake --build build --target all
