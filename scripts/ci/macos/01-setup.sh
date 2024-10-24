#!/bin/bash

brew install cmake
brew install xsimd openblas
brew install sentencepiece
brew install python-setuptools

sysctl -a | grep machdep.cpu.features

ulimit -c unlimited # Enable core dumps to be captured (must be in same run block)

COREDUMP_DIR="$PWD/slimt-coredump"
mkdir -p "${COREDUMP_DIR}"
COREDUMP_PATTERN="${COREDUMP_DIR}/core.%n.%P.%t"
sudo sysctl -w kern.corefile=${COREDUMP_PATTERN}

echo "coredumps: ${COREDUMP_PATTERN}"
