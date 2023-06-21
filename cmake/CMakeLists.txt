cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
project(xsdnn VERSION 0.0.0 LANGUAGES C CXX)

option(xsdnn_BUILD_TEST OFF)
option(xsdnn_USE_DOUBLE OFF)
option(xsdnn_USE_DETERMENISTIC_GEN OFF)
option(xsdnn_BUILD_SHARED OFF)
option(xsdnn_USE_SSE OFF)
option(xsdnn_USE_OPENMP OFF)

set(XSROOT ${PROJECT_SOURCE_DIR}/..)
set(XSROOT_SRC ${XSROOT}/src)
set(MMPACK_ROOT ${XSROOT_SRC}/mmpack/lib)
set(THIRD_PARTY_ROOT ${XSROOT}/cmake/external)

include_directories(
        ${XSROOT}/include/
)

if (xsdnn_USE_DOUBLE)
    add_compile_definitions(MM_USE_DOUBLE)
endif (xsdnn_USE_DOUBLE)

if (NOT xsdnn_USE_DETERMENISTIC_GEN)
    add_compile_definitions(XS_NO_DTRMNST)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

find_package(OpenMP QUIET)
if(xsdnn_USE_OPENMP AND OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    add_compile_definitions(XS_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
elseif(xsdnn_USE_OPENMP AND NOT OPENMP_FOUND)
    message(FATAL_ERROR "Can't find OpenMP. Please set OpenMP_C_FLAGS & "
            "OpenMP_CXX_FLAGS")
endif()

# Set flags to build type
set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing")
set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pg -fsanitize=address -fsanitize=leak -O0")

# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} -O3 -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

if (xsdnn_USE_SSE)
    add_compile_definitions(MM_USE_SSE)
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -msse3")
endif()

include(xsdnn_mmpack.cmake)
include(xsdnn_common.cmake)
include(xsdnn_core.cmake)
include(xsdnn_layers.cmake)
include(xsdnn_activations.cmake)
include(xsdnn_optimizers.cmake)
include(xsdnn_loss.cmake)
include(xsdnn_utils.cmake)

add_library(xsdnn STATIC
        ${mmpack_common_src}
        ${xsdnn_common_src}
        ${xsdnn_core_common_src}
        ${xsdnn_core_framework_src}
        ${xsdnn_core_kernel_src}
        ${xsdnn_layers_src}
        ${xsdnn_activations_src}
        ${xsdnn_optimizers_src}
        ${xsdnn_loss_src}
        ${xsdnn_utils_src})

if (xsdnn_BUILD_TEST)
    include(xsdnn_unittest.cmake)
endif (xsdnn_BUILD_TEST)

add_executable(simple_test ${XSROOT}/test/simple_test.cc)
target_link_libraries(simple_test GTest::gtest_main xsdnn)