//
// Created by rozhin on 24.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//
#include <core/kernel/activation_fwd.h>
#include <core/framework/threading.h>
using namespace mmpack;

#define TEMPLATE_ACTIVATION_PREPARATION                                     \
gsl::span<const float> XSpan = GetDataAsSpan<const float>(&X[sample]);      \
gsl::span<float> YSpan = GetMutableDataAsSpan<float>(&Y[sample]);           \
std::copy(X[sample].begin(), X[sample].end(), Y[sample].begin());

namespace xsdnn {
    namespace core {

void ComputeReluKernelFP32(const tensor_t& X,
                           tensor_t& Y,
                           params::activation& params,
                           bool parallelize,
                           size_t nthreads) {
    concurrency::TryParallelFor(parallelize,
                                nthreads,
                                X.size(),
                                [&](size_t sample) {
        TEMPLATE_ACTIVATION_PREPARATION
        size_t spatial_size = params.shape.size();
        MmActivation(&params.params, YSpan.data(), 1, spatial_size, spatial_size);
    });
}

void ComputeHardSigmoidKernelFP32(const tensor_t& X,
                                  tensor_t& Y,
                                  params::activation& params,
                                  bool parallelize,
                                  size_t nthreads) {
    concurrency::TryParallelFor(parallelize,
                                nthreads,
                                X.size(),
                                [&](size_t sample) {
        TEMPLATE_ACTIVATION_PREPARATION
        size_t spatial_size = params.shape.size();
        MmActivation(&params.params, YSpan.data(), 1, spatial_size, spatial_size);
    });
}

void ActivationFwdKernel::Compute(core::OpContext &ctx, params::activation &p) {
    const tensor_t& X = ctx.input_data(0);
    tensor_t& Y = ctx.output_data(0);

    MmActivationType AType = p.params.ActivationType;
    core::backend_t engine = ctx.engine();
    xsDtype dtype = ctx.dtype();

    if (engine == backend_t::xs) {
        if (AType == mmpack::Relu) {
            if (dtype == xsDtype::kXsFloat32) {
                ComputeReluKernelFP32(X, Y, p, ctx.parallelize(), ctx.num_threads());
            } else {
                throw xs_error("[activation Relu forward] unsupported dtype");
            }
        } else if (AType == mmpack::HardSigmoid) {
            if (dtype == xsDtype::kXsFloat32) {
                ComputeHardSigmoidKernelFP32(X, Y, p, ctx.parallelize(), ctx.num_threads());
            } else {
                throw xs_error("[activation HardSigmoid forward] unsupported dtype");
            }
        } else {
            throw xs_error("[activation forward] unsupported activation type");
        }
    } else {
        throw xs_error("[activation forward] unsupported engine type");
    }
}

    }
}