//
// Created by rozhin on 23.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/conv/conv_fwd_kernel.h>
#include <core/kernel/conv/conv_fwd_xs_impl.h>

namespace xsdnn {
    namespace core {

void ConvFwdKernel::compute(xsdnn::core::OpContext &ctx, params::conv &p) {
    const tensor_t& X = ctx.input_data(0);
    const tensor_t& W = ctx.input_data(1);
    const mat_t* B = p._.Bias ? &ctx.input_data(2).at(0) : nullptr;

    tensor_t& Y = ctx.output_data(0);

    backend_t engine = ctx.engine();

    if (engine == backend_t::xs) {
        kernel::conv_fwd_xs_impl(X, W[0], B, Y, p, ctx.parallelize(), ctx.num_threads());
    } else {
        xs_error("[conv forward] unsupported engine type");
    }
}

    } // core
} // xsdnn