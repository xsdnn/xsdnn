//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/linear/fully_connected_bwd_xs_impl.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace kernel {

void fully_connected_bwd_xs_impl(const tensor_t& x,
                                 const mat_t& W,
                                 tensor_t& dx,
                                 tensor_t& dW,
                                 tensor_t& db,
                                 tensor_t& dLz,
                                 const params::fully& p,
                                 bool parallelize,
                                 size_t nthreads) {
    size_t sample_count = x.size();
    size_t in_size = p.in_size_;
    size_t out_size = p.out_size_;

    mm_scalar alpha = 1.0f;
    mm_scalar beta  = 0.0f;

    concurrency::TryParallelFor(parallelize, nthreads, sample_count, [&](size_t sample) {
        /*
         * grad(x) = dLz * W.T [for each elem in batch]
         */
        mm_scalar* dLz_ptr = dLz[sample].data(); // A
        const mm_scalar* W_ptr = W.data();       // B
        mm_scalar* dx_ptr = dx[sample].data();   // C

        for (size_t i = 0; i < in_size; ++i) {
            dx_ptr[i] = mmpack::MmDot(dLz_ptr,
                                      &W_ptr[i * out_size],
                                      out_size);
        }

        /*
         * grad(W) = x.T * dLz [for each elem in batch]
         */

        dLz_ptr = dLz[sample].data();
        const mm_scalar* x_ptr = x[sample].data();
        mm_scalar* dW_ptr = dW[sample].data();

        mmpack::MmGemm(
                mmpack::CblasNoTrans,
                mmpack::CblasNoTrans,
                in_size, out_size, 1,
                alpha,
                x_ptr, 1,
                dLz_ptr, out_size,
                beta,
                dW_ptr, out_size
        );

        /*
         * grad(b) = dLz [for each elem in batch]
         */
        if (!db.empty()) {
            for (size_t i = 0; i < out_size; ++i) {
                db[sample][i] += dLz[sample][i];
            }
        }
    });
}

    } // kernel
} // xsdnn
