//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/linear/fully_connected_bwd_kernel.h>
#include <core/kernel/linear/fully_connected_bwd_xs_impl.h>
#include <utils/xs_error.h>

namespace xsdnn {
    namespace core {

void FullyConnectedBwdKernel::compute(xsdnn::core::OpContext &ctx, params::fully &p) {
    const tensor_t& x = ctx.input_data(0);
    const tensor_t& W = ctx.input_data(1);
    tensor_t& dW = ctx.input_grad(1);
    tensor_t* db = p.has_bias_ ? &ctx.input_grad(2) : nullptr;
    tensor_t& dx = ctx.input_grad(0);
    tensor_t& dLz = ctx.output_grad(0);
    tensor_t fake_tensor;

    bool paralellize = ctx.parallelize();
    core::backend_t engine = ctx.engine();

    if (engine == core::backend_t::xs) {
        kernel::fully_connected_bwd_xs_impl(x,
                                            W[0],
                                            dx,
                                            dW,
                                            p.has_bias_ ? *db : fake_tensor,
                                            dLz,
                                            p,paralellize, ctx.num_threads());
    } else {
        throw xs_error("Unsupported engine type");
    }
}

    } // core
} // xsdnn
