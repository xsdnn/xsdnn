cmake_minimum_required(VERSION 3.27.0)
SET(CMAKE_CXX_STANDARD 17)
PROJECT(xsdnn VERSION 1.0.0 LANGUAGES C CXX)

OPTION(xsdnn_BUILD_TEST OFF)
OPTION(xsdnn_USE_DETERMENISTIC_GEN OFF)
OPTION(xsdnn_USE_SSE OFF)
OPTION(xsdnn_USE_OPENMP OFF)
OPTION(xsdnn_BUILD_XNNPACK_ENGINE OFF)
OPTION(xsdnn_WITH_SERIALIZATION OFF)

SET(XSROOT ${PROJECT_SOURCE_DIR})
SET(XSROOT_SRC ${XSROOT}/src)
SET(MMPACK_ROOT ${XSROOT_SRC}/mmpack)
SET(THIRD_PARTY_ROOT ${XSROOT}/cmake/external)

INCLUDE_DIRECTORIES(
        ${XSROOT}/include/
)

IF (xsdnn_USE_DOUBLE)
    ADD_COMPILE_DEFINITIONS(MM_USE_DOUBLE)
ENDIF (xsdnn_USE_DOUBLE)

IF (NOT xsdnn_USE_DETERMENISTIC_GEN)
    ADD_COMPILE_DEFINITIONS(XS_NO_DTRMNST)
ENDIF()

IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(CMAKE_VERBOSE_MAKEFILE ON)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")

FIND_PACKAGE(OpenMP QUIET)
IF(xsdnn_USE_OPENMP AND OPENMP_FOUND)
    MESSAGE(STATUS "Found OpenMP")
    ADD_COMPILE_DEFINITIONS(XS_USE_OMP)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSEIF(xsdnn_USE_OPENMP AND NOT OPENMP_FOUND)
    MESSAGE(FATAL_ERROR "Can't find OpenMP. Please SET OpenMP_C_FLAGS & "
            "OpenMP_CXX_FLAGS")
ENDIF()

IF (xsdnn_USE_SSE)
    # Проверим поддержку sse в компиляторе. Supported only gcc.
    IF(CMAKE_COMPILER_IS_GNUCXX)
        INCLUDE(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
        IF(xsdnn_USE_SSE AND COMPILER_HAS_SSE_FLAG)
            ADD_DEFINITIONS(-DMM_USE_SSE)
            SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
            SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -msse3")
        ENDIF()
    ENDIF()

ENDIF()

#   ---[Download Deps]
IF (xsdnn_BUILD_XNNPACK_ENGINE)
    IF(NOT DEFINED XNNPACK_SOURCE_DIR)
        MESSAGE(STATUS "Downloading XNNPACK to ${CMAKE_BINARY_DIR}/xnnpack-source (define XNNPACK_SOURCE_DIR to avoid it)")
        CONFIGURE_FILE(./cmake/DownloadXNNPACK.cmake "${CMAKE_BINARY_DIR}/xnnpack-download/CMakeLists.txt")
        EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/xnnpack-download")
        EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/xnnpack-download")
        SET(XNNPACK_SOURCE_DIR "${CMAKE_BINARY_DIR}/xnnpack-source" CACHE STRING "xnnpack source directory")
    ENDIF()
    INCLUDE_DIRECTORIES("${XNNPACK_SOURCE_DIR}/include")
    INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/pthreadpool-source/include")
    ADD_COMPILE_DEFINITIONS(XS_USE_XNNPACK)

    IF(NOT TARGET xnnpack)
        ADD_SUBDIRECTORY(
                "${XNNPACK_SOURCE_DIR}"
                "${CMAKE_BINARY_DIR}/deps/xnnpack" EXCLUDE_FROM_ALL)
    ENDIF()
ENDIF()

IF (xsdnn_BUILD_TEST)
    IF(NOT DEFINED GOOGLETEST_SOURCE_DIR)
        MESSAGE(STATUS "Downloading GTest to ${CMAKE_BINARY_DIR}/googletest-source (define GOOGLETEST_SOURCE_DIR to avoid it)")
        CONFIGURE_FILE(./cmake/DownloadGoogleTest.cmake "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
        EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
        EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
        SET(GOOGLETEST_SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source" CACHE STRING "googletest source directory")
    ENDIF()
    INCLUDE_DIRECTORIES("${GOOGLETEST_SOURCE_DIR}/googletest/include")

    IF(NOT TARGET gtest)
        ADD_SUBDIRECTORY(
                "${GOOGLETEST_SOURCE_DIR}"
                "${CMAKE_BINARY_DIR}/deps/googletest" EXCLUDE_FROM_ALL)
    ENDIF()
ENDIF()

IF(xsdnn_WITH_SERIALIZATION)
    EXECUTE_PROCESS(COMMAND which protoc OUTPUT_VARIABLE PROTOC_INSTALLED OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF (PROTOC_INSTALLED MATCHES "protoc")
        SET(Protobuf_PROTOC_EXECUTABLE ${PROTOC_INSTALLED})
        SET(Protobuf_INCLUDE_DIR /usr/local/include)
        FIND_PACKAGE(Protobuf REQUIRED)
        INCLUDE_DIRECTORIES(${Protobuf_INCLUDE_DIRS})
        # ---[Compile model format]
        EXECUTE_PROCESS(COMMAND protoc -I ${XSROOT}/include/serializer/ --cpp_out=${XSROOT}/include/serializer/ xs.proto3)
        ADD_COMPILE_DEFINITIONS(XS_USE_SERIALIZATION)
    ELSE()
        MESSAGE(FATAL_ERROR "Install Protobuf Compiler C++ at System to make serialization available")
    ENDIF()
ENDIF()

# SET flags to build type
SET(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing")
SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
SET(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pg -fsanitize=address -fsanitize=leak")

# SET compiler options
SET(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} -O3 -ffast-math")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${EXTRA_C_FLAGS_DEBUG} -ffast-math")
SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} -O0")


INCLUDE(cmake/xsdnn_mmpack.cmake)
INCLUDE(cmake/xsdnn_common.cmake)
INCLUDE(cmake/xsdnn_core.cmake)
INCLUDE(cmake/xsdnn_layers.cmake)
INCLUDE(cmake/xsdnn_activations.cmake)
INCLUDE(cmake/xsdnn_utils.cmake)
INCLUDE(cmake/xsdnn_serializer.cmake)
INCLUDE(cmake/xsdnn_session.cmake)

IF(xsdnn_WITH_SERIALIZATION)
    INCLUDE(cmake/xsdnn_serializer.cmake)
ELSE()
    SET(xsdnn_serializer_src "")
ENDIF()

ADD_LIBRARY(xsdnn
        ${mmpack_common_src}
        ${xsdnn_common_src}
        ${xsdnn_core_common_src}
        ${xsdnn_core_framework_src}
        ${xsdnn_core_kernel_src}
        ${xsdnn_layers_src}
        ${xsdnn_activations_src}
        ${xsdnn_utils_src}
        ${xsdnn_serializer_src}
        ${xsdnn_session_src})


SET_TARGET_PROPERTIES(xsdnn PROPERTIES VERSION ${PROJECT_VERSION})

IF (xsdnn_BUILD_TEST)
    INCLUDE(cmake/xsdnn_unittest.cmake)
ENDIF (xsdnn_BUILD_TEST)

INCLUDE(cmake/summary.cmake)
xsdnn_print_configuration_summary()
