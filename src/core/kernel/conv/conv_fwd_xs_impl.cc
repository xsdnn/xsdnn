//
// Created by rozhin on 24.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/conv/conv_fwd_xs_impl.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace kernel {

void conv_fwd_xs_impl(const tensor_t& X,
                      const mat_t& W,
                      const mat_t* B,
                      tensor_t& Y,
                      params::conv& p,
                      bool parallelize,
                      size_t nthreads) {
    concurrency::TryParallelFor(parallelize, nthreads, X.size(), [&](size_t sample) {
        mat_t TemporaryBuffer(p._.TemproraryBufferSize);

        if (B != nullptr) {
            mmpack::MmConv(&p._,
                           X[sample].data(), W.data(), B->data(),
                           TemporaryBuffer.data(), Y[sample].data());
        } else {
            mmpack::MmConv(&p._,
                           X[sample].data(), W.data(), nullptr,
                           TemporaryBuffer.data(), Y[sample].data());
        }

        // Compute activation if there is
        if (p.activation_type_ != MmActivationType::NotSet) {
            MmActivationHolder ActHolder;
            ActHolder.ActivationType = p.activation_type_;
            MmSetDefaultActivationParameters(&ActHolder);

            MmActivation(&ActHolder, Y[sample].data(), p._.OutShape[0], p._.OutShape[1], p._.OutShape[1]);
        }
    });
}

    } // kernel
} // xsdnn