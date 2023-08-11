#!/bin/bash

set -eo pipefail

# Configure
cmake -B build -S $PWD -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --target all
