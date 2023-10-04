#!/bin/bash

set -eo pipefail

function cmake-configure {
  NDK=android-ndk-r23b
  ABI="arm64-v8a"
  MINSDK_VERSION=28
  ANDROID_PLATFORM=android-28

  mkdir -p build
  pushd build

  SLIMT_ARGS=(
    -DWITH_RUY=ON
    -DWITH_INTGEMM=OFF
    -DWITH_BLAS=OFF
    -DSLIMT_USE_INTERNAL_PCRE2=ON
  )

  OTHER_ANDROID_ARGS=(
    -DANDROID_ARM_NEON=TRUE
  )
  # Additionally list variables finally configured.
  set -x
  cmake -L \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_TOOLCHAIN=clang \
    -DANDROID_ABI=$ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DANDROID_NATIVE_API_LEVEL=$MINSDKVERSION \
    -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.8 \
    -DANDROID_STL=c++_static \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    "${SLIMT_ARGS[@]}" \
    "${OTHER_ANDROID_ARGS[@]}" \
    ..
  set +x
  popd
}

cmake-configure
cmake --build build --target all
