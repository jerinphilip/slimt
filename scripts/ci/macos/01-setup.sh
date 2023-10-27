#!/bin/bash

brew install cmake
brew install xsimd openblas
brew install sentencepiece

sysctl -a | grep machdep.cpu.features
