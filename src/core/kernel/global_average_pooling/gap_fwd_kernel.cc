//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/global_average_pooling/gap_fwd_kernel.h>
#include <core/kernel/global_average_pooling/gap_fwd_xs_impl.h>

namespace xsdnn {
    namespace core {

void GlobalAvgPoolingFwdKernel::compute(xsdnn::core::OpContext &ctx, params::global_avg_pool &p) {
    const BTensor & in_data = ctx.input_data(0);
    BTensor & out_data = ctx.output_data(0);

    for (size_t sample = 0; sample < out_data.size(); ++sample) {
        gsl::span<float> OutSpan = out_data[sample].GetMutableDataAsSpan<float>();
        std::fill(OutSpan.begin(), OutSpan.end(), 0.0f);
    }

    backend_t engine = ctx.engine();

    if (engine == backend_t::xs) {
        kernel::global_average_pool_fwd_xs_impl(in_data, out_data, p,
                                                ctx.parallelize(), ctx.num_threads());
    } else {
        xs_error("[global_average_pool forward] unsupported engine type");
    }
}

    } // core
} // xsdnn
