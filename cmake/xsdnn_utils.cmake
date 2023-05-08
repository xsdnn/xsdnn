set(
        xsdnn_utils_src
        "${XSROOT_SRC}/utils/color_print.h"
        "${XSROOT_SRC}/utils/macro.h"
        "${XSROOT_SRC}/utils/random.h"
        "${XSROOT_SRC}/utils/random.cc"
        "${XSROOT_SRC}/utils/rng.h"
        "${XSROOT_SRC}/utils/rng.cc"
        "${XSROOT_SRC}/utils/tensor.h"
        "${XSROOT_SRC}/utils/tensor_shape.h"
        "${XSROOT_SRC}/utils/tensor_shape.cc"
        "${XSROOT_SRC}/utils/tensor_utils.h"
        "${XSROOT_SRC}/utils/tensor_utils.cc"
        "${XSROOT_SRC}/utils/util.h"
        "${XSROOT_SRC}/utils/util.cc"
        "${XSROOT_SRC}/utils/weight_init.h"
        "${XSROOT_SRC}/utils/weight_init.cc"
        "${XSROOT_SRC}/utils/xs_error.h"
        "${XSROOT_SRC}/utils/xs_error.cc"
        "${XSROOT_SRC}/utils/grad_checker.h"
        "${XSROOT_SRC}/utils/grad_checker.cc"
)

add_library(xsdnn_utils STATIC ${xsdnn_utils_src})