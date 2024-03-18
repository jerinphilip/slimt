# Depending on the value of SLIMT_USE_INTERNAL_PRCRE2 this cmake file either
# tries to find the Perl Compatible Regular Expresison library (pcre2) on the
# system (when OFF), or downloads and compiles them locally (when ON).

# The following variables are set: PCRE2_FOUND - System has the PCRE library
# PCRE2_LIBRARIES - The PCRE library file PCRE2_INCLUDE_DIR - The folder with
# the PCRE headers

if(SLIMT_USE_INTERNAL_PCRE2)
  include(ExternalProject)

  set(PCRE2_VERSION "10.43")
  set(PCRE2_FILENAME "pcre2-${PCRE2_VERSION}")
  set(PCRE2_TARBALL "${PCRE2_FILENAME}.tar.gz")
  set(PCRE2_SOURCE_DIR "${CMAKE_BINARY_DIR}/${PCRE2_FILENAME}")

  # Download tarball only if we don't have the pcre2 source code yet. For the
  # time being, we download and unpack pcre2 into the ssplit source tree. This
  # is not particularly clean but allows us to wipe the build dir without having
  # to re-download pcre2 so often. Git has been instructed to ignore
  # ${PCRE2_SOURCE_DIR} via .gitignore.
  if(EXISTS ${PCRE2_SOURCE_DIR}/configure)
    set(PCRE2_URL "")
  else()
    set(PCRE2_URL
        "https://github.com/PCRE2Project/pcre2/releases/download/${PCRE2_FILENAME}/${PCRE2_TARBALL}"
    )
    message("Downloading pcre2 source code from ${PCRE2_URL}")
  endif()

  # Set configure options for internal pcre2 depeding on compiler
  if(CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$")
    # jit compilation isn't supported by wasm
    set(PCRE2_SUPPORT_JIT "OFF")
  else()
    set(PCRE2_SUPPORT_JIT "ON")
  endif()

  # CMAKE_CROSSCOMPILING_EMULATOR might contain semicolon (;) separated
  # arguments. Preventing list expansion on the arguments of this variable
  # before adding it to PCRE2_CONFIGURE_OPTIONS by replacing semicolon with
  # $<SEMICOLON> as per:
  # https://cmake.org/cmake/help/git-stage/manual/cmake-generator-expressions.7.html#genex:SEMICOLON
  string(REPLACE ";" "$<SEMICOLON>"
                 CMAKE_CROSSCOMPILING_EMULATOR_WITH_SEMICOLON
                 "${CMAKE_CROSSCOMPILING_EMULATOR}")

  include(GNUInstallDirs)
  set(PCRE2_CONFIGURE_OPTIONS
      -DBUILD_SHARED_LIBS=OFF
      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
      -DCMAKE_BUILD_TYPE=Release
      -DPCRE2_SUPPORT_JIT=${PCRE2_SUPPORT_JIT}
      # Necessary for proper MacOS (emscripten) compilation
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_CROSSCOMPILING_EMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR_WITH_SEMICOLON}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=true # Added for pybind11
  )

  # Android platform needs to be explicitly passed given this is an external
  # project. If not supplied armv8-a switches into armv7-a, making the compiled
  # library incompatible with an upstream bergamot-translator.
  if(ANDROID)
    list(APPEND PCRE2_CONFIGURE_OPTIONS -DANDROID_PLATFORM=${ANDROID_PLATFORM}
         -DANDROID_ABI=${ANDROID_ABI})
  endif(ANDROID)

  # set include dirs and libraries for PCRE2
  set(PCRE2_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
  set(PCRE2_LIBDIR ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
  set(PCRE2_LIBRARIES
      ${PCRE2_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}pcre2-8${CMAKE_STATIC_LIBRARY_SUFFIX}
  )

  # download, configure, compile
  ExternalProject_Add(
    pcre2
    PREFIX ${CMAKE_BINARY_DIR}
    URL ${PCRE2_URL}
    DOWNLOAD_DIR ${PCRE2_SOURCE_DIR}
    SOURCE_DIR ${PCRE2_SOURCE_DIR}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} ${PCRE2_SOURCE_DIR}
                      ${PCRE2_CONFIGURE_OPTIONS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_DIR ${CMAKE_BINARY_DIR})

  add_library(pcre2-lib INTERFACE)
  add_dependencies(pcre2-lib pcre2)
  set_target_properties(
    pcre2-lib PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PCRE2_INCLUDE_DIR}"
                         INTERFACE_LINK_LIBRARIES "${PCRE2_LIBRARIES}")
  add_library(PCRE2::PCRE2 ALIAS pcre2-lib)

else(SLIMT_USE_INTERNAL_PCRE2)
  find_library(
    PCRE2_LIBRARIES
    NAMES pcre2 pcre2-8 # shared
          pcre2-8-static pcre2-posix-static # static
          pcre2-8-staticd pcre2-posix-staticd # windows?
  )

  find_path(PCRE2_INCLUDE_DIR pcre2.h)
  if(PCRE2_LIBRARIES AND PCRE2_INCLUDE_DIR)
    mark_as_advanced(PCRE2_FOUND PCRE2_INCLUDE_DIR PCRE2_LIBRARIES
                     PCRE2_VERSION)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
      PCRE2
      REQUIRED_VARS PCRE2_INCLUDE_DIR PCRE2_LIBRARIES
      VERSION_VAR PCRE2_VERSION)
    set(PCRE2_FOUND TRUE)
  else()
    set(PCRE2_FOUND FALSE)
  endif()

  if(PCRE2_FOUND AND NOT TARGET PCRE2::PCRE2)
    add_library(PCRE2::PCRE2 INTERFACE IMPORTED)
    set_target_properties(
      PCRE2::PCRE2
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PCRE2_INCLUDE_DIR}"
                 INTERFACE_LINK_LIBRARIES "${PCRE2_LIBRARIES}")
  endif()
endif(SLIMT_USE_INTERNAL_PCRE2)
