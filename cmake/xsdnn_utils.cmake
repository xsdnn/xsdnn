set(
        xsdnn_utils_src
        "${XSROOT_SRC}/utils/random.cc"
        "${XSROOT_SRC}/utils/rng.cc"
        "${XSROOT_SRC}/utils/tensor_shape.cc"
        "${XSROOT_SRC}/utils/tensor_utils.cc"
        "${XSROOT_SRC}/utils/util.cc"
        "${XSROOT_SRC}/utils/weight_init.cc"
        "${XSROOT_SRC}/utils/xs_error.cc"
        "${XSROOT_SRC}/utils/grad_checker.cc"
)

add_library(xsdnn_utils STATIC ${xsdnn_utils_src})