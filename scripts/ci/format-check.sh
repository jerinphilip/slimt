#!/bin/bash

set -eo pipefail
set -x

function slimt-check-clang-format {
  # clang-format
  python3 run-clang-format.py --style file -r app slimt
}

function slimt-check-clang-tidy {
  # clang-tidy
  mkdir -p build
  pushd build
  cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    ..
  popd
  run-clang-tidy -p build "$PWD/slimt/.*"
  run-clang-tidy -p build "$PWD/app/.*"

}

function slimt-check-python {
  python3 -m black --diff --check scripts/
  python3 -m isort --profile black --diff --check scripts/
}

function slimt-check-sh {
  shfmt -i 2 -ci -bn -sr -d scripts/
}

function slimt-check-cmake {
  cmake-format $(find -name "CMakeLists.txt" -not -path "./3rd-party/*") --check
}

slimt-check-clang-format
slimt-check-python
slimt-check-sh
slimt-check-cmake
slimt-check-clang-tidy
