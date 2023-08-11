//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/batch_norm/bn_bwd_kernel.h>
#include <core/kernel/batch_norm/bn_bwd_xs_impl.h>

namespace xsdnn {
    namespace core {

    void BatchNormalizationBwdKernel::compute(xsdnn::core::OpContext &ctx, params::bnorm &p) {
        const tensor_t& in = ctx.input_data(0);
        const tensor_t& out = ctx.output_data(0);

        tensor_t& dx = ctx.input_grad(0);
        tensor_t& dg = ctx.input_grad(1);
        tensor_t& db = ctx.input_grad(2);
        const tensor_t& dlz = ctx.output_grad(1);

        backend_t engine = ctx.engine();

        if (engine == backend_t::xs) {
            kernel::batch_normalization_bwd_xs_impl(in, out, dx, dg, db, dlz, p,
                                                    ctx.parallelize(), ctx.num_threads());
        } else {
            xs_error("[batch_norm backward] unsupported engine type");
        }

    }

    } // core
} // xsdnn
