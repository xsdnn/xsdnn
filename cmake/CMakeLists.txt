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
)

#set (xsdnn_utils
#        ${XSROOT}/src/utils/random.cc
#        ${XSROOT}/src/utils/rng.cc
#        ${XSROOT}/src/utils/tensor_shape.cc
#        ${XSROOT}/src/utils/tensor_utils.cc
#        ${XSROOT}/src/utils/util.cc
#        ${XSROOT}/src/utils/weight_init.cc
#        ${XSROOT}/src/utils/xs_error.cc
#        )
#
#add_library(xsdnn_utils STATIC ${xsdnn_utils})
#
#set (xsdnn_common
#        ${XSROOT}/src/common/node.cc
#        ${XSROOT}/src/common/nodes.cc
#        ${XSROOT}/src/common/network.cc
#        )
#
#add_library(xsdnn_common STATIC ${xsdnn_common})
#
#set(xsdnn_layers
#        ${XSROOT}/src/layers/layer.cc
#        ${XSROOT}/src/layers/fully_connected.cc
#        )
#
#add_library(xsdnn_layers STATIC ${xsdnn_layers})
#
#set(xsdnn_core
#        ${XSROOT}/src/core/backend.cc
#        ${XSROOT}/src/core/framework/op_context.cc
#        ${XSROOT}/src/core/framework/op_kernel.cc
#        ${XSROOT}/src/core/kernel/fully_connected_fwd_kernel.cc
#        ${XSROOT}/src/core/kernel/fully_connected_fwd_xs_impl.cc
#        )
#
#add_library(xsdnn_core STATIC ${xsdnn_core})

set (xsdnn
        ${XSROOT}/src/utils/random.cc
        ${XSROOT}/src/utils/rng.cc
        ${XSROOT}/src/utils/tensor_shape.cc
        ${XSROOT}/src/utils/tensor_utils.cc
        ${XSROOT}/src/utils/util.cc
        ${XSROOT}/src/utils/weight_init.cc
        ${XSROOT}/src/utils/xs_error.cc
        ${XSROOT}/src/core/backend.cc
        ${XSROOT}/src/common/node.cc
        ${XSROOT}/src/common/nodes.cc
        ${XSROOT}/src/common/network.cc
        ${XSROOT}/src/layers/layer.cc
        ${XSROOT}/src/layers/fully_connected.cc
        ${XSROOT}/src/layers/layer_register.cc
        ${XSROOT}/src/core/framework/op_context.cc
        ${XSROOT}/src/core/framework/op_kernel.cc
        ${XSROOT}/src/core/kernel/fully_connected_fwd_kernel.cc
        ${XSROOT}/src/core/kernel/fully_connected_fwd_xs_impl.cc
        )
set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -pg")
add_library(xsdnn STATIC ${xsdnn})
