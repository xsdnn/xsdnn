//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/linear/fully_connected_fwd_kernel.h>
#include <core/kernel/linear/fully_connected_fwd_xs_impl.h>
#include <utils/tensor_utils.h>
#include <utils/xs_error.h>

namespace xsdnn {
    namespace core {

void FullyConnectedFwdKernel::compute(OpContext &ctx, params::fully &p) {
    const tensor_t& in = ctx.input_data(0);
    const tensor_t& W = ctx.input_data(1);
    const tensor_t* b = p.has_bias_ ? &ctx.input_data(2) : nullptr;
    tensor_t& out = ctx.output_data(0);

    core::backend_t engine = ctx.engine();

    if (engine == core::backend_t::xs) {
        kernel::fully_connected_fwd_xs_impl(in,
                                            W[0],
                                            p.has_bias_ ? (*b)[0] : mat_t(),
                                            out, p, ctx.parallelize(), ctx.num_threads());
    } else {
        throw xs_error("[fully_connected forward] unsuported engine type");
    }
}

    } // xsdnn
} // xsdnn