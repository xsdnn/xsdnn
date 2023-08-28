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
            throw xs_error("[conv fwd] without bias ot implemented yet");
//            mmpack::MmConv(&p._,
//                           X[sample].data(), W.data(), B->data(),
//                           TemporaryBuffer.data(), Y[sample].data());
        } else {
            mmpack::MmConv(&p._,
                           X[sample].data(), W.data(), nullptr,
                           TemporaryBuffer.data(), Y[sample].data());
        }
    });
}

    } // kernel
} // xsdnn