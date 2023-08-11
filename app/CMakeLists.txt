add_executable(slimt_main main.cc)
set_target_properties(slimt_main PROPERTIES OUTPUT_NAME "slimt")
target_link_libraries(slimt_main PUBLIC slimt)

add_executable(slimt_test test.cc)
set_target_properties(slimt_test PROPERTIES OUTPUT_NAME "slimt_test")
target_link_libraries(slimt_test PUBLIC slimt)
