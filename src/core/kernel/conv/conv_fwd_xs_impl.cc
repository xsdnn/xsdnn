//
// Created by rozhin on 24.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/conv/conv_fwd_xs_impl.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace kernel {

void conv_fwd_xs_impl(const BTensor & X,
                      const tensor_t& W,
                      const tensor_t* B,
                      BTensor & Y,
                      params::conv& p,
                      bool parallelize,
                      size_t nthreads) {
    concurrency::TryParallelFor(parallelize, nthreads, X.size(), [&](size_t sample) {
        tensor_t TemporaryBuffer(X[0].dtype(), p._.TemproraryBufferSize, nullptr);

        if (B != nullptr) {
            mmpack::MmConv(&p._,
                           X[sample].GetData<float>(), W.GetData<float>(), B->GetData<float>(),
                           TemporaryBuffer.GetMutableData<float>(), Y[sample].GetMutableData<float>());
        } else {
            mmpack::MmConv(&p._,
                           X[sample].GetData<float>(), W.GetData<float>(), nullptr,
                           TemporaryBuffer.GetMutableData<float>(), Y[sample].GetMutableData<float>());
        }

        // Compute activation if there is
        if (p.activation_type_ != mmpack::MmActivationType::NotSet) {
            mmpack::MmActivationHolder ActHolder;
            ActHolder.ActivationType = p.activation_type_;
            MmSetDefaultActivationParameters(&ActHolder);

            MmActivation(&ActHolder, Y[sample].GetMutableData<float>(), p._.OutShape[0], p._.OutShape[1], p._.OutShape[1]);
        }
    });
}

    } // kernel
} // xsdnn