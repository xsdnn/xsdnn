//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "fully_connected_bwd_xs_impl.h"

namespace xsdnn {
    namespace kernel {

void fully_connected_bwd_xs_impl(const tensor_t& x,
                                 const mat_t& W,
                                 tensor_t& dx,
                                 tensor_t& dW,
                                 tensor_t& db,
                                 tensor_t& dLz,
                                 const params::fully& p,
                                 bool parallelize) {
    XS_UNUSED_PARAMETER(parallelize);

    size_t sample_count = x.size();
    size_t in_size = p.in_size_;
    size_t out_size = p.out_size_;

    mm_scalar alpha = 1.0f;
    mm_scalar beta  = 0.0f;

    /*
     * grad(x) = dLz * W.T [for each elem in batch]
     */
    for (size_t sample = 0; sample < sample_count; ++sample) {
        mm_scalar* dLz_ptr = dLz[sample].data(); // A
        const mm_scalar* W_ptr = W.data();       // B
        mm_scalar* dx_ptr = dx[sample].data();   // C

        mmpack::MmGemm(mmpack::CblasNoTrans,
                       mmpack::CblasTrans,
                       1, out_size, in_size,
                       alpha,
                       dLz_ptr, out_size,
                       W_ptr, out_size,
                       beta,
                       dx_ptr, in_size);
    }

    /*
     * grad(W) = x.T * dLz [for each elem in batch]
     */
    for (size_t sample = 0; sample < sample_count; ++sample) {
        mm_scalar* dLz_ptr = dLz[sample].data();
        const mm_scalar* x_ptr = x[sample].data();
        mm_scalar* dW_ptr = dW[sample].data();

        mmpack::MmGemm(mmpack::CblasTrans,
                       mmpack::CblasNoTrans,
                       1, out_size, in_size,
                       alpha,
                       x_ptr, in_size,
                       dLz_ptr, out_size,
                       beta,
                       dW_ptr, out_size);
    }

    /*
     * grad(b) = dLz [for each elem in batch]
     */
    // FIXME : добавить векторизацию через simd
    if (!db.empty()) {
        for (size_t sample = 0; sample < sample_count; ++sample) {
            for (size_t i = 0; i < out_size; ++i) {
                db[sample][i] += dLz[sample][i];
            }
        }
    }
}

    } // kernel
} // xsdnn
