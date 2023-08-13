#!/bin/bash
#
#

function cmake-configure {
  NDK=/opt/android-ndk
  ABI="arm64-v8a"
  MINSDK_VERSION=28
  ANDROID_PLATFORM=android-28

  mkdir -p build
  pushd build
  OTHER_ANDROID_ARGS=(
    -DANDROID_ARM_NEON=TRUE
  )
  OTHER_MARIAN_ARGS=(
    -DCMAKE_HAVE_THREADS_LIBRARY=1
    -DCMAKE_USE_WIN32_THREADS_INIT=0
    -DCMAKE_USE_PTHREADS_INIT=1
    -DTHREADS_PREFER_PTHREAD_FLAG=ON
    # -DCOMPILE_WITHOUT_EXCEPTIONS=on # Apparently this can reduce the binary size, let's see.
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
    -DWITH_INTGEMM=OFF \
    -DWITH_BLAS=OFF \
    -DANDROID_STL=c++_static \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    "${OTHER_ANDROID_ARGS[@]}" "${OTHER_MARIAN_ARGS[@]}" \
    ..
  set +x
  popd
}

cmake-configure
cmake --build build --target all
