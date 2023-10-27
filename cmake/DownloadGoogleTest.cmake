CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0 FATAL_ERROR)

PROJECT(googletest-download NONE)

# Set file timestamps to the time of extraction.
IF(POLICY CMP0135)
  CMAKE_POLICY(SET CMP0135 NEW)
ENDIF()

# LINT.IfChange
INCLUDE(ExternalProject)
ExternalProject_Add(googletest
  URL https://github.com/google/googletest/archive/e23cdb78e9fef1f69a9ef917f447add5638daf2a.zip
  URL_HASH SHA256=5cb522f1427558c6df572d6d0e1bf0fd076428633d080e88ad5312be0b6a8859
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source"
  BINARY_DIR "${CMAKE_BINARY_DIR}/googletest"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)