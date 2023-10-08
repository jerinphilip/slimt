
message("Using Python: " ${Python_EXECUTABLE})

# pybind11 method:
pybind11_add_module(_slimt SHARED slimt.cpp)
target_link_libraries(_slimt PUBLIC pybind11::module pybind11::headers slimt)
# target_include_directories(_slimt PUBLIC ${PROJECT_SOURCE_DIR}
# ${PROJECT_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src)
