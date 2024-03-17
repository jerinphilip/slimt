find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

add_library(slimt_jni SHARED slimt.cpp)

target_link_libraries(slimt_jni slimt)
