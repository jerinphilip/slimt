#!/bin/bash

set -eo pipefail

(cd build && cpack -G DEB)
