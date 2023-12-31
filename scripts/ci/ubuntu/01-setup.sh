#!/bin/bash

set -x
sudo apt-get update
sudo apt-get install -y build-essential cmake

XSIMD_VERSION="11.1.0"
wget https://github.com/xtensor-stack/xsimd/archive/refs/tags/${XSIMD_VERSION}.tar.gz -O xsimd.tar.gz
tar xf xsimd.tar.gz

(
  cd xsimd-${XSIMD_VERSION} \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/ -B build -S . \
    && cmake --build build --target all && sudo make -C build install
)

lscpu

# Prepare core-dumps
COREDUMP_DIR="$PWD/slimt-coredump"
mkdir -p "${COREDUMP_DIR}"
COREDUMP_PATTERN="${COREDUMP_DIR}/core-%e-%p-%t"

echo "${COREDUMP_PATTERN}" | sudo tee /proc/sys/kernel/core_pattern

echo "coredumps: ${COREDUMP_PATTERN}"

python3 -m pip install setuptools
