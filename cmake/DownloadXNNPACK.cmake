CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0 FATAL_ERROR)

PROJECT(xnnpack-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
    CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

INCLUDE(ExternalProject)
ExternalProject_Add(xnnpack
        URL https://github.com/google/XNNPACK/archive/refs/heads/master.zip
        SOURCE_DIR "${CMAKE_BINARY_DIR}/xnnpack-source"
        BINARY_DIR "${CMAKE_BINARY_DIR}/xnnpack"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
        CMAKE_ARGS -DXNNPACK_BUILD_BENCHMARKS=OFF -DXNNPACK_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        )