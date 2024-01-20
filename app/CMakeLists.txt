add_executable(slimt_cli main.cc)
set_target_properties(slimt_cli PROPERTIES OUTPUT_NAME "slimt-cli")
target_link_libraries(slimt_cli PUBLIC slimt)

set(SLIMT_BINARIES slimt_cli)

if(UNIX)
  install(TARGETS ${SLIMT_BINARIES} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif(UNIX)
