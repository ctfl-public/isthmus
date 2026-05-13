# Python extension target for the isthmus library.
# Invoked when ISTHMUS_BUILD_PYTHON=ON or via scikit-build-core (SKBUILD set).

pybind11_add_module(_isthmus isthmus_binding.cpp)

target_link_libraries(_isthmus PRIVATE isthmus_cpp::isthmus_cpp)

# scikit-build-core installs CMake artifacts relative to the wheel root.
# Placing _isthmus inside the `isthmus` sub-directory aligns with the
# Python package layout in python/isthmus/.
install(TARGETS _isthmus DESTINATION isthmus)

# For manual CMake builds (not via scikit-build-core / pip), copy the compiled
# extension into the source package directory so a simple sys.path.insert is
# enough to import isthmus without any installation step.
if(NOT DEFINED SKBUILD)
    add_custom_command(TARGET _isthmus POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:_isthmus>
            "${CMAKE_CURRENT_SOURCE_DIR}/isthmus/$<TARGET_FILE_NAME:_isthmus>"
        COMMENT "Copying _isthmus into python/isthmus/ for development use"
    )
endif()
