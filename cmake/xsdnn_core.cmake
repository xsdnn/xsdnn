set(
        xsdnn_core_common_src
        "${XSROOT_SRC}/core/backend.cc"
)

set(
        xsdnn_core_framework_src
        "${XSROOT_SRC}/core/framework/op_context.cc"
        "${XSROOT_SRC}/core/framework/op_kernel.cc"
)

set(
        xsdnn_core_kernel_src
        "${XSROOT_SRC}/core/kernel/linear/fully_connected_fwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/linear/fully_connected_bwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/linear/fully_connected_fwd_xs_impl.cc"
        "${XSROOT_SRC}/core/kernel/linear/fully_connected_bwd_xs_impl.cc"
        "${XSROOT_SRC}/core/kernel/batch_norm/bn_fwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/batch_norm/bn_bwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/batch_norm/bn_bwd_xs_impl.cc"
        "${XSROOT_SRC}/core/kernel/batch_norm/bn_fwd_xs_impl.cc"
)

add_library(xsdnn_core STATIC
        ${xsdnn_core_common_src}
        ${xsdnn_core_framework_src}
        ${xsdnn_core_kernel_src})
