//
// Created by rozhin on 20.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//
#include <core/kernel/max_pool_fwd.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace core {

void ComputeMaxPoolKernelFP32(const tensor_t& in_data,
                              tensor_t& out_data,
                              params::max_pool& p,
                              bool parallelize,
                              size_t nthreads) {
    concurrency::TryParallelFor(parallelize, nthreads, in_data.size(), [&](size_t sample){
        gsl::span<const float> InSpan = GetDataAsSpan<const float>(&in_data[sample]);
        gsl::span<float> OutSpan = GetMutableDataAsSpan<float>(&out_data[sample]);

        for (size_t i = 0; i < p.out2in.size(); ++i) {
            const auto& in_idx = p.out2in[i];
            mm_scalar max_value = std::numeric_limits<mm_scalar>::lowest();
            for (const auto& j : in_idx) {
                max_value = std::max(max_value, InSpan[j]);
            }
            OutSpan[i] = max_value;
        }
    });
}

        void MaxPoolingFwdKernel::Compute(core::OpContext &ctx, params::max_pool &p) {
            const tensor_t& in_data = ctx.input_data(0);
            tensor_t& out_data = ctx.output_data(0);

            backend_t engine = ctx.engine();
            xsDtype dtype = ctx.dtype();

            if (engine == backend_t::xs) {
                if (dtype == kXsFloat32) ComputeMaxPoolKernelFP32(in_data, out_data, p, ctx.parallelize(), ctx.num_threads());
                else throw xs_error("[max_pool fwd] unsupported dtype for xs engine");
            } else {
                throw xs_error("[max_pool forward] unsupported engine type");
            }
        }

    } // core
} // xsdnn