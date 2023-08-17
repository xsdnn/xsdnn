//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/global_average_pooling/gap_fwd_xs_impl.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace kernel {

void global_average_pool_fwd_xs_impl(const tensor_t& in_data,
                                     tensor_t& out_data,
                                     params::global_avg_pool& p,
                                     bool parallelize,
                                     size_t nthreads) {
    // TODO: make it for sse / avx
    concurrency::TryParallelFor(parallelize, nthreads, in_data.size(), [&](size_t sample) {
       const mat_t& in = in_data[sample];
       mat_t& out = out_data[sample];

       size_t spatial_size =  p.in_shape_.H * p.in_shape_.W;
       for (size_t c = 0; c < p.in_shape_.C; ++c) {
           for (size_t d = 0; d < spatial_size; ++d) {
                out[c] += in[c * spatial_size + d];
           }
           out[c] /= spatial_size;
       }

    });
}

    } // kernel
} // xsdnn