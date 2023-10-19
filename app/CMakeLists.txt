add_executable(slimt main.cc)
set_target_properties(slimt PROPERTIES OUTPUT_NAME "slimt")
target_link_libraries(slimt PUBLIC slimt-static)

add_executable(slimt_test test.cc)
set_target_properties(slimt_test PROPERTIES OUTPUT_NAME "slimt_test")
target_link_libraries(slimt_test PUBLIC slimt-static)

set(SLIMT_BINARIES slimt)

if(UNIX)
  install(TARGETS ${SLIMT_BINARIES} DESTINATION ${CMAKE_INSTALL_BINDIR})
  message(STATUS "Exports: ${SLIMT_BINARIES}")
endif(UNIX)
