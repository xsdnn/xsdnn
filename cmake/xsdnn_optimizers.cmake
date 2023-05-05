set(
        xsdnn_optimizers_src
        "${XSROOT_SRC}/optimizers/optimizer_base.h"
        "${XSROOT_SRC}/optimizers/optimizer_base.cc"
)

add_library(xsdnn_optimizers STATIC ${xsdnn_optimizers_src})