set(
        xsdnn_layers_src
        "${XSROOT_SRC}/layers/layer.h"
        "${XSROOT_SRC}/layers/layer.cc"
        "${XSROOT_SRC}/layers/fully_connected.h"
        "${XSROOT_SRC}/layers/fully_connected.cc"
        "${XSROOT_SRC}/layers/layer_register.cc"
)

add_library(xsdnn_layers STATIC ${xsdnn_layers_src})