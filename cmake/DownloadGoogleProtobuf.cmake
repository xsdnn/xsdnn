CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0 FATAL_ERROR)

PROJECT(protobuf-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
    CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(protobuf
        URL https://github.com/protocolbuffers/protobuf/releases/download/v24.4/protobuf-24.4.zip
        SOURCE_DIR "${CMAKE_BINARY_DIR}/protobuf-source"
        BINARY_DIR "${CMAKE_BINARY_DIR}/protobuf"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
        )