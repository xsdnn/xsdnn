//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/batch_norm/bn_fwd_kernel.h>
#include <core/kernel/batch_norm/bn_fwd_xs_impl.h>

namespace xsdnn {
    namespace core {



void BatchNormalizationFwdKernel::compute(xsdnn::core::OpContext &ctx, params::bnorm &p) {
    const tensor_t& in = ctx.input_data(0);
    const tensor_t& gamma = ctx.input_data(1);
    const tensor_t& beta = ctx.input_data(2);
    tensor_t& out = ctx.output_data(0);

    if (!p.statistic_initialized) {
        init_statistics(p);
    }

    backend_t engine = ctx.engine();

    if (engine == backend_t::xs) {
        kernel::batch_normalization_fwd_xs_impl(in, gamma[0], beta[0], out, p,
                                                ctx.parallelize(), ctx.num_threads());
    } else {
        xs_error("[batch_norm forward] unsupported engine type");
    }
}

void BatchNormalizationFwdKernel::init_statistics(params::bnorm &p) {
    size_t in_channels = p.in_shape_.C;

    p.stat_holder["mean_running_"] = mat_t(in_channels);
    p.stat_holder["stddev_running_"] = mat_t(in_channels);
    p.stat_holder["mean_"] = mat_t(in_channels);
    p.stat_holder["stddev_"] = mat_t(in_channels);

    p.statistic_initialized = true;
}

    } // core
} // xsdnn