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

void XNNPACKComputeConvKernelFP32(const tensor_t& X,
                                  const mat_t& W,
                                  const mat_t* B,
                                  tensor_t& Y,
                                  params::conv& p,
                                  bool parallelize,
                                  size_t nthreads) {
    if (X.size() != 1) throw xs_error(START_MSG + "Support only batch == 1 at XNNPACK backend engine");
    if (nthreads != 1) throw xs_error(START_MSG + "Support only nthreads == 1 at XNNPACK backend engine");
    const size_t padding_top = p._.Padding[0];
    const size_t padding_left = p._.Padding[1];
    const size_t padding_bottom = p._.Padding[2];
    const size_t padding_right = p._.Padding[3];
    const size_t kernel_h = p._.KernelShape[0];
    const size_t kernel_w = p._.KernelShape[1];
    const size_t stride_h = p._.StrideShape[0];
    const size_t stride_w = p._.StrideShape[1];
    const size_t dilation_h = p._.DilationShape[0];
    const size_t dilation_w = p._.DilationShape[1];
    const size_t groups = p._.GroupCount;
    const size_t group_input_channels = p._.InChannel;
    const size_t group_output_channels = p._.FilterCount; // FIXME: неуверен

    const size_t output_pixel_stride = groups * group_output_channels;
    const size_t input_pixel_stride = groups * group_input_channels;

    gsl::span<const float> kernel = GetDataAsSpan<const float>(&W);
    gsl::span<const float> bias;
    if (B != nullptr) {
        bias = GetDataAsSpan<const float>(&W);
    }

    xnn_operator_t ConvolutionOp;
    xnn_status status;

    status = xnn_create_convolution2d_nchw_f32(padding_top, padding_right, padding_bottom, padding_left,
                                               kernel_h, kernel_w,
                                               stride_h, stride_w,
                                               dilation_h, dilation_w,
                                               groups, group_input_channels, group_output_channels,
                                               input_pixel_stride, output_pixel_stride,
                                               kernel.data(), bias.data(),
                                               -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
                                               0 /* flags */, nullptr, nullptr, &ConvolutionOp);

    // TODO: Try make it for ARM?
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
        else throw xs_error(START_MSG + "unsupported dtype for xs engine");
    } else if (engine == backend_t::xnnpack) {
        if (dtype == kXsFloat32) XNNPACKComputeConvKernelFP32(X, W[0], B, Y, p, ctx.parallelize(), ctx.num_threads());
        else throw xs_error(START_MSG + "unsupported dtype for xnnpack engine");
    } else {
        throw xs_error(START_MSG + "unsupported engine type");
    }

}

    } // core
} // xsdnn