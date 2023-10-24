set(
        xsdnn_core_common_src
        "${XSROOT_SRC}/core/backend.cc"
)

set(
        xsdnn_core_framework_src
        "${XSROOT_SRC}/core/framework/op_context.cc"
        "${XSROOT_SRC}/core/framework/op_kernel.cc"
        "${XSROOT_SRC}/core/framework/params.cc"
)

set(
        xsdnn_core_kernel_src
        "${XSROOT_SRC}/core/kernel/activation_fwd.cc"
        "${XSROOT_SRC}/core/kernel/fully_connected_fwd.cc"
        "${XSROOT_SRC}/core/kernel/conv_fwd.cc"
        "${XSROOT_SRC}/core/kernel/global_avg_pool_fwd.cc"
        "${XSROOT_SRC}/core/kernel/max_pool_fwd.cc"
)
