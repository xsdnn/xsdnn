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
        "${XSROOT_SRC}/core/kernel/linear/fully_connected_fwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/linear/fully_connected_fwd_xs_impl.cc"
        "${XSROOT_SRC}/core/kernel/max_pool/mp_fwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/max_pool/mp_fwd_xs_impl.cc"
        "${XSROOT_SRC}/core/kernel/global_average_pooling/gap_fwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/global_average_pooling/gap_fwd_xs_impl.cc"
        "${XSROOT_SRC}/core/kernel/conv/conv_fwd_kernel.cc"
        "${XSROOT_SRC}/core/kernel/conv/conv_fwd_xs_impl.cc"
)
