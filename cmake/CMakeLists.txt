cmake_minimum_required(VERSION 3.13)
project(xsdnn CXX)


option(xsdnn_BUILD_TEST OFF)
option(xsdnn_USE_DOUBLE OFF)
option(xsdnn_USE_DETERMENISTIC_GEN OFF)

set(XSROOT ${PROJECT_SOURCE_DIR}/..)

if (xsdnn_USE_DOUBLE)
    add_compile_definitions(MM_USE_DOUBLE)
endif (xsdnn_USE_DOUBLE)

include_directories(
        external/
)

set(xsdnn_utils
        ${XSROOT}/src/utils/rng.cc
        ${XSROOT}/src/utils/random.cc
)

add_library(xsdnn_utils STATIC ${xsdnn_utils})

