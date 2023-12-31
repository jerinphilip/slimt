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
  ARGS=(
    -DCMAKE_EXPORT_COMPILE_COMMANDS=on
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    -DUSE_BUILTIN_SENTENCEPIECE=ON
    -DWITH_INTGEMM=ON -DUSE_AVX2=ON -DUSE_SSE2=ON

    # Gemmology, which is default on has to be turned off.
    -DWITH_GEMMOLOGY=OFF
  )

  cmake -B build -S . "${ARGS[@]}"
  set +e
  FILES=$(find app slimt -type f)
  run-clang-tidy -export-fixes build/clang-tidy.slimt.yml -fix -format -p build -header-filter="$PWD/slimt" ${FILES[@]}
  CHECK_STATUS=$?
  git diff
  set -e
  return $CHECK_STATUS

}

function slimt-check-python {
  python3 -m black --diff --check scripts/
  python3 -m isort --profile black --diff --check scripts/
}

function slimt-check-sh {
  shfmt -i 2 -ci -bn -sr -d scripts/
}

function slimt-check-cmake {
  set +e
  CMAKE_FILES=$(find -name "CMakeLists.txt" -not -path "./3rd-party/*" -not -path "build")
  cmake-format ${CMAKE_FILES[@]} --check
  CHECK_STATUS=$?
  set -e
  cmake-format ${CMAKE_FILES[@]} --in-place
  git diff
  return $CHECK_STATUS
}

function slimt-check-iwyu {
  iwyu-tool -p build slimt/* > build/iwyu.out
}

slimt-check-clang-format
slimt-check-python
slimt-check-sh
slimt-check-cmake
slimt-check-clang-tidy
