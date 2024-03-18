find_package(JNI REQUIRED)

add_library(slimt_jni SHARED slimt.cpp)
target_link_libraries(slimt_jni PRIVATE slimt-static JNI::JNI)
