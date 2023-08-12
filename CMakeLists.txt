cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(slimt CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WITH_ASAN "Enable AddressSanitizer in builds" OFF)
option(WITH_TESTS "Enable building tests (requires a few extra libraries)" OFF)

add_subdirectory(3rd-party)

find_package(BLAS)

set(SLIMT_EXTRA_COMPILE_OPTIONS -Wall -Wpedantic -Wextra -Wno-unknown-pragmas
                                -Werror)

if(FALSE)
  set(SLIMT_EXTERNAL_LIBS Sentencepiece::Sentencepiece intgemm BLAS::BLAS)
  set(SLIMT_EXTRA_COMPILE_OPTIONS ${SLIMT_EXTRA_COMPILE_OPTIONS} -march=native)
else(FALSE)
  set(SLIMT_EXTERNAL_LIBS Sentencepiece::Sentencepiece ruy)
endif(FALSE)

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
target_include_directories(slimt PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(slimt PUBLIC ${CMAKE_BINARY_DIR})
add_subdirectory(app)
