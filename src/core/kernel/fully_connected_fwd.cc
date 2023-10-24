//
// Created by rozhin on 19.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/fully_connected_fwd.h>
#include <core/framework/threading.h>
#include <cstring>

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
        gsl::span<const float> InSpan = GetDataAsSpan<const float>(&in[sample]);
        gsl::span<const float> WSpan = GetDataAsSpan<const float>(&W);
        gsl::span<float> OutSpan = GetMutableDataAsSpan<float>(&out[sample]);

        const float* in_ptr = InSpan.data();
        const float* w_ptr = WSpan.data();
        float* out_ptr = OutSpan.data();

        if (b.empty()) {
            memset(out_ptr, 0, sizeof(mm_scalar) * out_size);
        } else {
            gsl::span<const float> BSpan = GetDataAsSpan<const float>(&b); // FIXME: здесь все ок?
            memcpy(out_ptr, BSpan.data(), sizeof(float) * out_size);
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
        if (dtype == kXsFloat32) ComputeFullyConnectedKernelFP32(in,
                                                                 W[0],
                                                                 p.has_bias_ ? (*b)[0] : mat_t(),
                                                                 out, p, ctx.parallelize(), ctx.num_threads());
        else throw xs_error("[fully_connected forward] unsupported dtype for xs engine");
    } else {
        throw xs_error("[fully_connected forward] unsupported engine type");
    }
}

    }
}
