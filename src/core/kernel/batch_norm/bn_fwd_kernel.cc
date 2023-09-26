//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/batch_norm/bn_fwd_kernel.h>
#include <core/kernel/batch_norm/bn_fwd_xs_impl.h>

namespace xsdnn {
    namespace core {

void BatchNormalizationFwdKernel::compute(xsdnn::core::OpContext &ctx, params::bnorm &p) {
    const BTensor& in = ctx.input_data(0);
    const BTensor& gamma = ctx.input_data(1);
    const BTensor& beta = ctx.input_data(2);
    BTensor& out = ctx.output_data(0);

    if (!p.statistic_initialized) {
        init_statistics(p, in[0].dtype());
    }

    backend_t engine = ctx.engine();

    if (engine == backend_t::xs) {
        kernel::batch_normalization_fwd_xs_impl(in, gamma[0], beta[0], out, p,
                                                ctx.parallelize(), ctx.num_threads());
    } else {
        xs_error("[batch_norm forward] unsupported engine type");
    }
}

void BatchNormalizationFwdKernel::init_statistics(params::bnorm &p, XsDtype TensorDtype) {
    size_t in_channels = p.in_shape_.C;

    p.stat_holder["mean_running_"] = tensor_t(TensorDtype, in_channels, nullptr);
    p.stat_holder["stddev_running_"] = tensor_t(TensorDtype, in_channels, nullptr);
    p.stat_holder["mean_"] = tensor_t(TensorDtype, in_channels, nullptr);
    p.stat_holder["stddev_"] = tensor_t(TensorDtype, in_channels, nullptr);

    p.statistic_initialized = true;
}

    } // core
} // xsdnn