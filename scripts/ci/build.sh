#!/bin/bash

set -eo pipefail

# Configure
cmake -B build -S $PWD -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build --target all
