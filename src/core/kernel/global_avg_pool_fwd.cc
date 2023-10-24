//
// Created by rozhin on 20.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/global_avg_pool_fwd.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace core {

void ComputeGlobalAvgPoolKernelFP32(const tensor_t& in_data,
                                    tensor_t& out_data,
                                    params::global_avg_pool& p,
                                    bool parallelize,
                                    size_t nthreads) {
    // TODO: Make it for sse
    concurrency::TryParallelFor(parallelize, nthreads, in_data.size(), [&](size_t sample) {
        gsl::span<const float> InSpan = GetDataAsSpan<const float>(&in_data[sample]);
        gsl::span<float> OutSpan = GetMutableDataAsSpan<float>(&out_data[sample]);

        size_t spatial_size =  p.in_shape_.H * p.in_shape_.W;
        for (size_t c = 0; c < p.in_shape_.C; ++c) {
            for (size_t d = 0; d < spatial_size; ++d) {
                OutSpan[c] += InSpan[c * spatial_size + d];
            }
            OutSpan[c] /= spatial_size;
        }
    });
}

void GlobalAvgPoolingFwdKernel::Compute(xsdnn::core::OpContext &ctx, params::global_avg_pool &p) {
    const tensor_t& in_data = ctx.input_data(0);
    tensor_t& out_data = ctx.output_data(0);

    backend_t engine = ctx.engine();
    xsDtype dtype = ctx.dtype();

    if (engine == backend_t::xs) {
        if (dtype == kXsFloat32) ComputeGlobalAvgPoolKernelFP32(in_data, out_data, p, ctx.parallelize(), ctx.num_threads());
        else throw xs_error("[global_avg_pool fwd] unsupported dtype for xs engine");
    } else {
        throw xs_error("[global_average_pool fwd] unsupported engine type");
    }
}

    } // core
} // xsdnn