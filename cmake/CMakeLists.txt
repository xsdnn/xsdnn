cmake_minimum_required(VERSION 3.13)
project(xsdnn CXX)


option(xsdnn_BUILD_TEST OFF)
option(xsdnn_USE_DOUBLE OFF)
option(xsdnn_USE_DETERMENISTIC_GEN OFF)

set(XSROOT ${PROJECT_SOURCE_DIR}/..)

if (xsdnn_USE_DOUBLE)
    add_compile_definitions(MM_USE_DOUBLE)
endif (xsdnn_USE_DOUBLE)

if (NOT xsdnn_USE_DETERMENISTIC_GEN)
    add_compile_definitions(XS_NO_DTRMNST)
endif()

include_directories(
        external/
        external/mmpack/cmake/external/pector/include
)

set (xsdnn_utils
        ${XSROOT}/src/utils/random.cc
        ${XSROOT}/src/utils/rng.cc
        ${XSROOT}/src/utils/tensor_shape.cc
        ${XSROOT}/src/utils/tensor_utils.cc
        ${XSROOT}/src/utils/util.cc
        ${XSROOT}/src/utils/weight_init.cc
        ${XSROOT}/src/utils/xs_error.cc
        )

add_library(xsdnn_utils STATIC ${xsdnn_utils})

set (xsdnn_nodes
        ${XSROOT}/src/node.cc
        ${XSROOT}/src/nodes.cc
        )

add_library(xsdnn_nodes STATIC ${xsdnn_nodes})

set(xsdnn_layers
        ${XSROOT}/src/layers/layer.cc)

add_library(xsdnn_layers STATIC ${xsdnn_layers})

