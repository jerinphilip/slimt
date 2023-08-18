cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(slimt CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WITH_ASAN "Enable AddressSanitizer in builds" OFF)
option(WITH_TESTS "Enable building tests (requires a few extra libraries)" OFF)

add_subdirectory(3rd-party)

set(SLIMT_COMPILE_OPTIONS -Wall -Wpedantic -Wextra -Wno-unknown-pragmas -Werror)
set(SLIMT_COMPILE_DEFINITIONS "")

option(WITH_INTGEMM "Use intgemm" ON)
option(WITH_RUY "Use ruy" OFF)
option(WITH_BLAS "Use BLAS. Otherwise moves to ruy" ON)

set(SLIMT_EXTERNAL_LIBS Sentencepiece::Sentencepiece)

if(WITH_BLAS)
  find_package(BLAS)
  set(SLIMT_EXTERNAL_LIBS ${SLIMT_EXTERNAL_LIBS} BLAS::BLAS)
  set(SLIMT_COMPILE_DEFINITIONS ${SLIMT_COMPILE_DEFINITIONS} SLIMT_HAS_BLAS)
  if(APPLE)
    set(BLAS_VENDOR "Accelerate")
    # see https://developer.apple.com/documentation/accelerate for more info you
    # may need to install Xcode command line tools if you don't have them
    # already (https://developer.apple.com/xcode/features/)
    add_library(apple_accelerate INTERFACE)
    target_link_libraries(apple_accelerate INTERFACE "-framework Accelerate")
    # https://stackoverflow.com/a/58304566
    target_include_directories(
      apple_accelerate
      INTERFACE
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
    )
    add_library(AppleLibs::accelerate ALIAS apple_accelerate)
    set(SLIMT_EXTERNAL_LIBS ${SLIMT_EXTERNAL_LIBS} AppleLibs::accelerate)
  endif(APPLE)
endif(WITH_BLAS)

if(WITH_INTGEMM)
  set(SLIMT_EXTERNAL_LIBS ${SLIMT_EXTERNAL_LIBS} intgemm)
  set(SLIMT_COMPILE_DEFINITIONS ${SLIMT_COMPILE_DEFINITIONS} SLIMT_HAS_INTGEMM)
  set(SLIMT_COMPILE_OPTIONS ${SLIMT_COMPILE_OPTIONS} -march=native)
endif(WITH_INTGEMM)

if(WITH_RUY)
  set(SLIMT_COMPILE_DEFINITIONS ${SLIMT_COMPILE_DEFINITIONS} SLIMT_HAS_RUY)
  set(SLIMT_EXTERNAL_LIBS ${SLIMT_EXTERNAL_LIBS} ruy)
endif(WITH_RUY)

# cmake-format: off
set(CMAKE_CXX_FLAGS_PROFILE           "${CMAKE_CXX_FLAGS_RELEASE} -pg" CACHE STRING "Flags used by the C++ compiler during profile builds." FORCE)
set(CMAKE_C_FLAGS_PROFILE             "${CMAKE_C_FLAGS_RELEASE} -pg" CACHE STRING "Flags used by the C compiler during profile builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -pg" CACHE STRING "Flags used for linking binaries during profile builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}-pg" CACHE STRING "Flags used by the shared libraries linker during profile builds." FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_PROFILE 
  CMAKE_C_FLAGS_PROFILE 
  CMAKE_EXE_LINKER_FLAGS_PROFILE
  CMAKE_SHARED_LINKER_FLAGS_PROFILE
)
# cmake-format: on

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(slimt)
target_include_directories(slimt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(slimt PUBLIC ${CMAKE_BINARY_DIR})
add_subdirectory(app)
