CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0 FATAL_ERROR)

PROJECT(xnnpack-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
    CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

INCLUDE(ExternalProject)
ExternalProject_Add(xnnpack
        URL https://github.com/yse/easy_profiler/archive/refs/heads/develop.zip
        SOURCE_DIR "${CMAKE_BINARY_DIR}/profiler-source"
        BINARY_DIR "${CMAKE_BINARY_DIR}/profiler"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
        )