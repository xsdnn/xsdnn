//
// Created by rozhin on 19.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/fully_connected_fwd.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace core {

void ComputeFullyConnectedKernelFP32(const tensor_t& in,
                                     const mat_t& W,
                                     const mat_t& b,
                                     tensor_t& out,
                                     const params::fully& p,
                                     bool parallelize,
                                     size_t nthreads) {
    size_t in_size = p.in_size_;
    size_t out_size = p.out_size_;
    float alpha = 1.0;
    float beta = 1.0;

    concurrency::TryParallelFor(parallelize, nthreads, in.size(), [&](size_t sample) {
        gsl::span<const float> InSpan = GetDataAsSpan<float>(&in[sample]);
        const mm_scalar* in_ptr = in[sample].data();
        const mm_scalar* w_ptr = W.data();
        mm_scalar* out_ptr = out[sample].data();

        if (b.empty()) {
            memset(out_ptr, 0, sizeof(mm_scalar) * out_size);
        } else {
            memcpy(out_ptr, b.data(), sizeof(mm_scalar) * out_size);
        }

        mmpack::MmGemm(mmpack::CblasNoTrans,
                       mmpack::CblasNoTrans,
                       1, out_size, in_size,
                       alpha,
                       in_ptr, in_size,
                       w_ptr, out_size,
                       beta,
                       out_ptr, out_size);
    });
}

void FullyConnectedFwdKernel::Compute(xsdnn::core::OpContext &ctx, params::fully &p) {
    const tensor_t& in = ctx.input_data(0);
    const tensor_t& W = ctx.input_data(1);
    const tensor_t* b = p.has_bias_ ? &ctx.input_data(2) : nullptr;
    tensor_t& out = ctx.output_data(0);

    core::backend_t engine = ctx.engine();
    xsDtype dtype = ctx.dtype();

    if (engine == core::backend_t::xs) {
        if (dtype == kXsFloat32) ComputeFullyConnectedKernelFP32();
        throw xs_error("[fully_connected forward] unsupported dtype for xs engine");
    } else {
        throw xs_error("[fully_connected forward] unsupported engine type");
    }
}

    }
}
