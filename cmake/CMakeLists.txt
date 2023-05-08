cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
project(xsdnn VERSION 0.0.0 LANGUAGES C CXX)

option(xsdnn_BUILD_TEST OFF)
option(xsdnn_USE_DOUBLE OFF)
option(xsdnn_USE_DETERMENISTIC_GEN OFF)
option(xsdnn_BUILD_SHARED OFF)

set(XSROOT ${PROJECT_SOURCE_DIR}/..)
set(XSROOT_SRC ${XSROOT}/src)
set(MMPACK_ROOT ${XSROOT_SRC}/mmpack/lib)

if (xsdnn_USE_DOUBLE)
    add_compile_definitions(MM_USE_DOUBLE)
endif (xsdnn_USE_DOUBLE)

if (NOT xsdnn_USE_DETERMENISTIC_GEN)
    add_compile_definitions(XS_NO_DTRMNST)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

# Set flags to build type
set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing")
set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pg -fsanitize=address -O0")

# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

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
