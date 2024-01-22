# Tests require generation Add a custom command to be executed during the build
if(SLIMT_GENERATED_UNIT_TESTS)
  set(SLIMT_TEST_UNIT "${CMAKE_CURRENT_BINARY_DIR}/generated-units.cc")
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_custom_command(
    OUTPUT ${SLIMT_TEST_UNIT}
    COMMAND
      "/usr/bin/python3" "${CMAKE_CURRENT_SOURCE_DIR}/generate-units.py" #
      "--trace" "${CMAKE_SOURCE_DIR}/traces/ml-xlit.trace.txt" #
      "--output" "${SLIMT_TEST_UNIT}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate-units.py"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(slimt_test_units ${SLIMT_TEST_UNIT})
  target_link_libraries(slimt_test_units PUBLIC slimt)
  target_include_directories(slimt_test_units
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()
