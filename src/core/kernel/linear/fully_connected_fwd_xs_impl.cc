//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/linear/fully_connected_fwd_xs_impl.h>
#include <core/framework/threading.h>
#include <cstring>

namespace xsdnn {
    namespace kernel {

void fully_connected_fwd_xs_impl(const tensor_t& in,
                                 const mat_t& W,
                                 const mat_t& b,
                                 tensor_t& out,
                                 const params::fully& p,
                                 bool parallelize,
                                 size_t nthreads) {

    size_t in_size = p.in_size_;
    size_t out_size = p.out_size_;
    mm_scalar alpha = 1.0;
    mm_scalar beta = 1.0;

    concurrency::TryParallelFor(parallelize, nthreads, in.size(), [&](size_t sample) {
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

    } // kernel
} // xsdnn