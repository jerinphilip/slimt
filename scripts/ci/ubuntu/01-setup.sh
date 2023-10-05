#!/bin/bash

set -x
sudo apt-get update
sudo apt-get install -y build-essential cmake
# sudo apt-get install -y libxsimd-dev

git clone https://github.com/xtensor-stack/xsimd
(cd xsimd && cmake -DCMAKE_INSTALL_PREFIX=/usr/ && make install)
