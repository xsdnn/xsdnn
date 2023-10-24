//
// Created by rozhin on 20.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/conv_fwd.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace core {

void ComputeConvKernelFP32(const tensor_t& X,
                           const mat_t& W,
                           const mat_t* B,
                           tensor_t& Y,
                           params::conv& p,
                           bool parallelize,
                           size_t nthreads) {
    concurrency::TryParallelFor(parallelize, nthreads, X.size(), [&](size_t sample) {
        mat_t TemporaryBuffer(p._.TemproraryBufferSize); // TODO: make it like weights
        gsl::span<const float> InSpan = GetDataAsSpan<const float>(&X[sample]);
        gsl::span<const float> WSpan = GetDataAsSpan<const float>(&W);
        gsl::span<float> TmpBufferSpan = GetMutableDataAsSpan<float>(&TemporaryBuffer);
        gsl::span<float> OutSpan = GetMutableDataAsSpan<float>(&Y[sample]);

        if (B != nullptr) {
            gsl::span<const float> BSpan = GetDataAsSpan<const float>(B);
            mmpack::MmConv(&p._,
                           InSpan.data(), WSpan.data(), BSpan.data(),
                           TmpBufferSpan.data(), OutSpan.data());
        } else {
            mmpack::MmConv(&p._,
                           InSpan.data(), WSpan.data(), nullptr,
                           TmpBufferSpan.data(), OutSpan.data());
        }

        // Compute activation if there is
        if (p.activation_type_ != MmActivationType::NotSet) {
            MmActivationHolder ActHolder;
            ActHolder.ActivationType = p.activation_type_;
            MmSetDefaultActivationParameters(&ActHolder);

            MmActivation(&ActHolder, OutSpan.data(), p._.OutShape[0], p._.OutShape[1], p._.OutShape[1]);
        }
    });
}

void ConvFwdKernel::Compute(core::OpContext &ctx, params::conv &p) {
    const tensor_t& X = ctx.input_data(0);
    const tensor_t& W = ctx.input_data(1);
    const mat_t* B = p._.Bias ? &ctx.input_data(2).at(0) : nullptr;

    tensor_t& Y = ctx.output_data(0);

    backend_t engine = ctx.engine();
    xsDtype dtype = ctx.dtype();

    if (engine == backend_t::xs) {
        if (dtype == kXsFloat32) ComputeConvKernelFP32(X, W[0], B, Y, p, ctx.parallelize(), ctx.num_threads());
        else throw xs_error("[conv forward] unsupported dtype for xs engine");
    } else {
        throw xs_error("[conv forward] unsupported engine type");
    }
}

    } // core
} // xsdnn