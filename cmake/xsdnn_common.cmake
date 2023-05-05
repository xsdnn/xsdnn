set(
        xsdnn_common_src
        "${XSROOT_SRC}/common/config.h"
        "${XSROOT_SRC}/common/node.h"
        "${XSROOT_SRC}/common/node.cc"
        "${XSROOT_SRC}/common/nodes.h"
        "${XSROOT_SRC}/common/nodes.cc"
        "${XSROOT_SRC}/common/network.h"
        "${XSROOT_SRC}/common/network.cc"
)

add_library(xsdnn_common STATIC ${xsdnn_common_src})