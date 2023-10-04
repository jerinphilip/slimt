# SentencePiece
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(SPM_USE_BUILTIN_PROTOBUF ON)
add_subdirectory(sentencepiece EXCLUDE_FROM_ALL)

# Create an alias target for SentencePiece
add_library(spiece_iface INTERFACE)
target_link_libraries(spiece_iface INTERFACE sentencepiece-static)
get_target_property(SentencePiece_INCLUDES sentencepiece-static
                    INCLUDE_DIRECTORIES)
target_include_directories(spiece_iface INTERFACE ${SentencePiece_INCLUDES}
                                                  sentencepiece/src/)

add_library(SentencePiece::SentencePiece ALIAS spiece_iface)

# intgemm
set(POSITION_INDEPENDENT_CODE ON)
set(INTGEMM_DONT_BUILD_TESTS ON)
add_subdirectory(intgemm EXCLUDE_FROM_ALL)

# ruy
add_subdirectory(ruy EXCLUDE_FROM_ALL)
