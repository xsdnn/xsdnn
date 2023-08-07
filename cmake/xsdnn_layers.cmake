set(
        xsdnn_layers_src
        "${XSROOT_SRC}/layers/layer.cc"
        "${XSROOT_SRC}/layers/fully_connected.cc"
        "${XSROOT_SRC}/layers/input.cc"
        "${XSROOT_SRC}/layers/add.cc"
        "${XSROOT_SRC}/layers/abs.cc"
        "${XSROOT_SRC}/layers/acos.cc"
        "${XSROOT_SRC}/layers/and.cc"
        "${XSROOT_SRC}/layers/flatten.cc"
        "${XSROOT_SRC}/layers/batch_normalization.cc"
        "${XSROOT_SRC}/layers/layer_register.cc"
)

add_library(xsdnn_layers STATIC ${xsdnn_layers_src})