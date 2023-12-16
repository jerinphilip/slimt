message("Using Python: " ${PYTHON_EXECUTABLE})

# pybind11 method:
pybind11_add_module(_slimt SHARED slimt.cpp)

# Decide whether to link with static or shared using switch.
if(SLIMT_PYTHON_LINK_STATIC)
  set(SLIMT_LIB slimt-static)
else(SLIMT_PYTHON_LINK_STATIC)
  set(SLIMT_LIB slimt-shared)
endif(SLIMT_PYTHON_LINK_STATIC)

target_link_libraries(_slimt PUBLIC pybind11::module pybind11::headers
                                    ${SLIMT_LIB})
