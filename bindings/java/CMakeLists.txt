find_package(JNI REQUIRED)

add_library(slimt_jni SHARED slimt.cpp)
target_include_directories(slimt_jni PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(slimt_jni slimt)
