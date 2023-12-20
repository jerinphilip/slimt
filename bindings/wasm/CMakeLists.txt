
# See https://github.com/emscripten-core/emscripten/blob/main/src/settings.js
list(APPEND WASM_COMPILE_FLAGS
  -O3
  # Preserve whitespaces in JS even for release builds; this doesn't increase wasm binary size
  $<$<CONFIG:Release>:-g1>
  # Relevant Debug info only for release with debug builds as this increases wasm binary size
  $<$<CONFIG:RelWithDebInfo>:-g2>
  -fPIC
  -mssse3
  -msimd128
  # -fno-exceptions # Can't do that because spdlog uses exceptions
  -sDISABLE_EXCEPTION_CATCHING=1
  -sSTRICT=1
)
list(APPEND WASM_LINK_FLAGS
  -O3
  # Preserve whitespaces in JS even for release builds; this doesn't increase wasm binary size
  $<$<CONFIG:Release>:-g1>
  # Relevant Debug info only for release with debug builds as this increases wasm binary size
  $<$<CONFIG:RelWithDebInfo>:-g2>
  -lembind
  # Save some code, and some speed
  -sASSERTIONS=0
  -sDISABLE_EXCEPTION_CATCHING=1
  # the intgemm functions we call will be undefined since these are linked at
  # runtime by our own javascript.
  -sLLD_REPORT_UNDEFINED
  -sERROR_ON_UNDEFINED_SYMBOLS=0
  # Cause we can!
  -sSTRICT=1
  # You know we need it
  -sALLOW_MEMORY_GROWTH=1
  -sENVIRONMENT=web,worker
  # No need to call main(), there's nothing there.
  -sINVOKE_RUN=0
  # No need for filesystem code in the generated Javascript
  -sFILESYSTEM=0
  # If you turn this on, it will mangle names which makes the dynamic linking hard.
  -sDECLARE_ASM_MODULE_EXPORTS=0
  # Export all of the intgemm functions in case we need to fall back to using the embedded intgemm
  -sEXPORTED_FUNCTIONS=[_int8PrepareAFallback,_int8PrepareBFallback,_int8PrepareBFromTransposedFallback,_int8PrepareBFromQuantizedTransposedFallback,_int8PrepareBiasFallback,_int8MultiplyAndAddBiasFallback,_int8SelectColumnsOfBFallback]
  # Necessary for mozintgemm linking. This prepares the `wasmMemory` variable ahead of time as
  # opposed to delegating that task to the wasm binary itself. This way we can link MozIntGEMM
  # module to the same memory as the main bergamot-translator module.
  -sIMPORTED_MEMORY=1
  # Dynamic execution is either frowned upon or blocked inside browser extensions
  -sDYNAMIC_EXECUTION=0
)

add_executable(wasm-slimt slimt.cpp)

# Generate version file that can be included in the wasm artifacts
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/project_version.js.in
               ${CMAKE_CURRENT_BINARY_DIR}/project_version.js @ONLY)

target_link_libraries(wasm-slimt PRIVATE slimt-static)

# This compile definition is required for generating binding code properly
target_compile_definitions(wasm-slimt PRIVATE WASM_BINDINGS)
target_compile_options(wasm-slimt PRIVATE ${WASM_COMPILE_FLAGS})
target_link_options(wasm-slimt PRIVATE ${WASM_LINK_FLAGS})
target_link_options(wasm-slimt PRIVATE --extern-pre-js=${CMAKE_CURRENT_BINARY_DIR}/project_version.js)

set_target_properties(wasm-slimt PROPERTIES
                        OUTPUT_NAME "slimt"
                        SUFFIX ".js"
                        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                      )

target_link_libraries(wasm-slimt slimt)

