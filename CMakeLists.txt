cmake_minimum_required(VERSION 3.5.1...3.27)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(slimt CXX C)
set(PROJECT_DESCRIPTION "Inference frontend for bergamot models.")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBLITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINE_HIDDEN YES)

option(WITH_ASAN "Enable AddressSanitizer in builds" OFF)
option(WITH_TESTS "Enable building tests (requires a few extra libraries)" OFF)

option(WITH_INTGEMM "Use intgemm" OFF)
option(WITH_RUY "Use ruy" OFF)
option(WITH_GEMMOLOGY "Use gemmology" ON)
option(WITH_BLAS "Use BLAS. Otherwise moves to ruy" ON)

option(USE_BUILTIN_SENTENCEPIECE "Use SentencePiece supplied as 3rd-party" ON)

option(USE_AVX512 "Use AVX512" OFF)
option(USE_AVX2 "Use AVX2" OFF)
option(USE_SSSE3 "Use SSSE3" OFF)
option(USE_SSE2 "Use SSE2" OFF)
option(USE_NEON "Use NEON" OFF)

option(SLIMT_PACKAGE "Package for cmake, pkgconfig" OFF)

set(SLIMT_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wno-unknown-pragmas
    -Werror
    -Wno-missing-field-initializers
    -Wno-unused-function
    -pedantic
    # -pedantic fires `;` on older systems, not on new. Might as well remove.
    # See: https://stackoverflow.com/q/77254582/4565794
    -Wextra-semi)

set(SLIMT_PUBLIC_LIBS "")
set(SLIMT_INTERFACE_LIBS "")
set(SLIMT_PRIVATE_LIBS "")
set(SLIMT_COMPILE_DEFINITIONS "")
set(SLIMT_LINK_OPTIONS "")

if(WITH_ASAN)
  list(APPEND SLIMT_COMPILE_OPTIONS -fsanitize=address -fsanitize=undefined)
  list(APPEND SLIMT_LINK_OPTIONS -fsanitize=address -fsanitize=undefined)
endif(WITH_ASAN)

if(WITH_TCMALLOC)
  find_library(
    TCMALLOC_LIBRARIES
    NAMES tcmalloc_minimal
    PATHS /usr/lib /usr/local/lib)

  if(TCMALLOC_LIBRARIES)
    message(STATUS "tcmalloc found: ${TCMALLOC_LIBRARIES}")
  else(TCMALLOC_LIBRARIES)
    message(FATAL_ERROR "tcmalloc not found")
  endif(TCMALLOC_LIBRARIES)

  list(APPEND SLIMT_PRIVATE_LIBS ${TCMALLOC_LIBRARIES})
endif(WITH_TCMALLOC)

find_package(PCRE2 REQUIRED)
add_subdirectory(3rd-party)

list(APPEND SLIMT_PRIVATE_LIBS PCRE2::PCRE2)
list(APPEND SLIMT_PUBLIC_LIBS SentencePiece::SentencePiece)

if(NOT USE_BUILTIN_SENTENCEPIECE)
  list(APPEND SLIMT_PRIVATE_LIBS SentencePiece::Protobuf)
endif(NOT USE_BUILTIN_SENTENCEPIECE)

if(WITH_BLAS)
  find_package(BLAS REQUIRED)

  # The above defines BLAS::BLAS only one recent cmakes, the following code
  # detects if the target is available if not creates an alias target.
  if(BLAS_FOUND AND NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(
      BLAS::BLAS PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIR} "
                            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
  endif()

  list(APPEND SLIMT_PRIVATE_LIBS BLAS::BLAS)
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
    list(APPEND SLIMT_PRIVATE_LIBS AppleLibs::accelerate)
  endif(APPLE)
endif(WITH_BLAS)

if(WITH_INTGEMM)
  list(APPEND SLIMT_PRIVATE_LIBS intgemm)
  list(APPEND SLIMT_COMPILE_DEFINITIONS SLIMT_HAS_INTGEMM)
  list(APPEND SLIMT_COMPILE_OPTIONS -march=native)
endif(WITH_INTGEMM)

if(WITH_RUY)
  list(APPEND SLIMT_COMPILE_DEFINITIONS SLIMT_HAS_RUY)
  list(APPEND SLIMT_PRIVATE_LIBS ruy)
endif(WITH_RUY)

if(WITH_GEMMOLOGY)
  find_package(xsimd REQUIRED)
  add_library(gemmology INTERFACE)
  target_link_libraries(gemmology INTERFACE xsimd)
  target_include_directories(gemmology
                             INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party")
  list(APPEND SLIMT_PRIVATE_LIBS gemmology)
  list(APPEND SLIMT_COMPILE_DEFINITIONS SLIMT_HAS_GEMMOLOGY)
endif(WITH_GEMMOLOGY)

if(USE_AVX512)
  list(APPEND SLIMT_COMPILE_DEFINITIONS USE_AVX512)
endif(USE_AVX512)

if(USE_AVX2)
  list(APPEND SLIMT_COMPILE_DEFINITIONS USE_AVX2)
endif(USE_AVX2)

if(USE_SSE2)
  list(APPEND SLIMT_COMPILE_DEFINITIONS USE_SSE2)
endif(USE_SSE2)

if(USE_SSSE3)
  list(APPEND SLIMT_COMPILE_DEFINITIONS USE_SSSE3)
endif(USE_SSSE3)

if(USE_NEON)
  list(APPEND SLIMT_COMPILE_DEFINITIONS USE_NEON)
endif(USE_NEON)

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

message(STATUS "Project name: ${PROJECT_NAME}")
set(PROJECT_VERSION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/slimt.version)
include(GetVersionFromFile)
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")

if(UNIX)
  include(GNUInstallDirs)
  # cmake-format: off
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_PACKAGE_CONTACT "Jerin Philip <jerinphilip@live.in>")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jerin Philip")
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR}/deb)
  set(CPACK_GENERATOR "DEB")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS libsentencepiece-dev)
  set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
  set(CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME})
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION_STRING})
  set(CPACK_DEBIAN_PACKAGE_NAME ${PROJECT_NAME})
  set(CPACK_DEBIAN_PACKAGE_VERSION "${PROJECT_VERSION_STRING}")
  set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "slimt: Inference library for tiny translation models.")
  # cmake-format: on
  include(CPack)
endif(UNIX)

add_subdirectory(slimt)
add_subdirectory(app)

if(BUILD_PYTHON)
  if(USE_PYBIND11_SOURCE)
    message(STATUS "Using pybind11 from source.")
    add_subdirectory(3rd-party/pybind11)
  else(USE_PYBIND11_SOURCE)
    find_package(pybind11 REQUIRED)
  endif(USE_PYBIND11_SOURCE)

  add_subdirectory(bindings/python)
endif(BUILD_PYTHON)
