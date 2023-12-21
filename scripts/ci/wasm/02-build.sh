source /etc/profile.d/emscripten.sh

CONFIGURE_ARGS=(
  -DWITH_BLAS=OFF -DWITH_RUY=ON -DWITH_GEMMOLOGY=OFF -DSLIMT_USE_INTERNAL_PCRE2=ON -DBUILD_WASM=ON
  -DBUILD_SHARED=OFF
)

which emcc
which emcmake

emcc --version

BUILD_DIR=build/wasm

emcmake cmake ${CONFIGURE_ARGS[@]} -B ${BUILD_DIR} -S $PWD
cmake --build ${BUILD_DIR} --target clean
cmake --build ${BUILD_DIR} --target all
