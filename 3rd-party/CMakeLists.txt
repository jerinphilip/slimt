# Sentencepiece
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(SPM_USE_BUILTIN_PROTOBUF ON)
add_subdirectory(sentencepiece EXCLUDE_FROM_ALL)

# Create an alias target for Sentencepiece
add_library(spiece_iface INTERFACE)
target_link_libraries(spiece_iface INTERFACE sentencepiece-static)

get_target_property(Sentencepiece_INCLUDES sentencepiece-static
                    INCLUDE_DIRECTORIES)
target_include_directories(spiece_iface INTERFACE ${Sentencepiece_INCLUDES})

target_compile_options(spiece_iface INTERFACE -D_USE_BUILTIN_STRING_VIEW)
add_library(Sentencepiece::Sentencepiece ALIAS spiece_iface)

# intgemm
set(INTGEMM_DONT_BUILD_TESTS ON)
add_subdirectory(intgemm EXCLUDE_FROM_ALL)

# ruy
add_subdirectory(ruy EXCLUDE_FROM_ALL)

add_subdirectory(ssplit EXCLUDE_FROM_ALL)
