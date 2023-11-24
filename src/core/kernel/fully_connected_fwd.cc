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

#ifdef XS_USE_XNNPACK
void XNNPACKComputeFullyConnectedKernelFP32(const tensor_t& X,
                                  tensor_t& Y,
                                  params::fully& p,
                                  xnn_operator_t FullyConnectedOp,
                                  pthreadpool_t threadpool) {
    if (X.size() != 1) throw xs_error(START_MSG + "Support only batch == 1 at XNNPACK backend engine");
    xnn_status status;

    gsl::span<const float> XSpan = GetDataAsSpan<const float>(&X[0]);
    gsl::span<float> YSpan = GetMutableDataAsSpan<float>(&Y[0]);

    status = xnn_setup_fully_connected_nc_f32(FullyConnectedOp, XSpan.data(), YSpan.data());
    if (status != xnn_status_success) {
        throw xs_error(START_MSG + "Failed to setup FP32 XNNPACK Convolution operator. Code: " + std::to_string(status));
    }

    status = xnn_run_operator(FullyConnectedOp, /*threadpool=*/threadpool);
    if (status != xnn_status_success) {
        throw xs_error(START_MSG + "Failed to run FP32 XNNPACK Convolution operator. Code: " + std::to_string(status));
    }
}
#endif

#ifdef XS_USE_XNNPACK
void FullyConnectedFwdKernel::CreateAndReshapeXNNKernel(xsdnn::xsDtype dtype, std::vector<mat_t *> WB, params::fully &p) {
    assert(p.has_bias_ ? WB.size() == 2 : WB.size() == 1);
    size_t Cin = p.in_size_;
    size_t Cout = p.out_size_;
    gsl::span<const float> kernel = GetDataAsSpan<const float>(WB[0]);
    gsl::span<const float> bias;
    if (p.has_bias_) bias = GetDataAsSpan<const float>(WB[1]);

    xnn_status status;
    if (dtype == kXsFloat32) {
        status = xnn_create_fully_connected_nc_f32(Cin, Cout,
                                   Cin, Cout,
                                   kernel.data(), bias.data(),
                                   -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
                                    /*flags=*/0, nullptr, nullptr, &op_);

        if (status != xnn_status_success) {
            throw xs_error(START_MSG + "Error when creating XNNPACK fully_connected_nc_fp32 kernel. Code: " + std::to_string(status));
        }

        status = xnn_reshape_fully_connected_nc_f32(
                op_,
                /*batch_size=*/1,
                /*threadpool=*/concurrency::threadpool::getInstance().threadpool_);

        if (status != xnn_status_success) {
            throw xs_error(START_MSG + "Failed to reshape FP32 XNNPACK FullyConnected operator. Code:" + std::to_string(status));
        }
    } else {
        throw xs_error("Unsupported dtype for XNNPACK FullyConnected operator");
    }

}
#endif

void FullyConnectedFwdKernel::Compute(xsdnn::core::OpContext &ctx, params::fully &p) {
    const tensor_t& X = ctx.input_data(0);
    const tensor_t& W = ctx.input_data(1);
    const tensor_t* B = p.has_bias_ ? &ctx.input_data(2) : nullptr;
    tensor_t& Y = ctx.output_data(0);

    core::backend_t engine = ctx.engine();
    xsDtype dtype = ctx.dtype();

    if (engine == core::backend_t::xs) {
        if (dtype == kXsFloat32) ComputeFullyConnectedKernelFP32(X,
                                                                 W[0],
                                                                 p.has_bias_ ? (*B)[0] : mat_t(),
                                                                 Y, p, ctx.parallelize(), ctx.num_threads());
        else throw xs_error("[fully_connected forward] unsupported dtype for xs engine");
    } else if (engine == backend_t::xnnpack) {
#ifdef XS_USE_XNNPACK
        if (dtype == kXsFloat32) XNNPACKComputeFullyConnectedKernelFP32(X, Y, p,
                                                              op_, concurrency::threadpool::getInstance().threadpool_);
        else throw xs_error(START_MSG + "unsupported dtype for xnnpack engine");
#else
        throw xs_error(START_MSG + "This build doesn't support XNN Backend Engine. "
                                   "Rebuild with -Dxsdnn_BUILD_XNNPACK_ENGINE=ON");
#endif
    } else throw xs_error("[fully_connected forward] unsupported engine type");
}

    }
}
