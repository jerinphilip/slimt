#!/bin/bash

set -x
sudo apt-get update
sudo apt-get install -y build-essential cmake

XSIMD_VERSION="11.1.0"
wget https://github.com/xtensor-stack/xsimd/archive/refs/tags/${XSIMD_VERSION}.tar.gz -O xsimd.tar.gz
tar xf xsimd.tar.gz
(cd xsimd-${XSIMD_VERSION} && cmake -DCMAKE_INSTALL_PREFIX=/usr/ && sudo make install)

lscpu
