set(
        xsdnn_activations_src
        "${XSROOT_SRC}/layers/activations/activation_layer.h"
        "${XSROOT_SRC}/layers/activations/activation_layer.cc"
        "${XSROOT_SRC}/layers/activations/relu.h"
        "${XSROOT_SRC}/layers/activations/relu.cc"
)

add_library(xsdnn_activations STATIC ${xsdnn_activations_src})