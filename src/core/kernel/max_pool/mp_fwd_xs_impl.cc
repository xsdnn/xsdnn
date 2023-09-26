//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/max_pool/mp_fwd_xs_impl.h>
#include <core/framework/threading.h>

namespace xsdnn {
    namespace kernel {

void max_pool_fwd_xs_impl(const BTensor & in_data,
                          BTensor & out_data,
                          params::max_pool& p,
                          bool parallelize,
                          size_t nthreads) {
    concurrency::TryParallelFor(parallelize, nthreads, in_data.size(), [&](size_t sample){
        gsl::span<const float> in = in_data[sample].GetDataAsSpan<float>();
        gsl::span<float> out = out_data[sample].GetMutableDataAsSpan<float>();

        for (size_t i = 0; i < p.out2in.size(); ++i) {
            const auto& in_idx = p.out2in[i];
            float max_value = std::numeric_limits<float>::lowest();
            for (const auto& j : in_idx) {
                max_value = std::max(max_value, in[j]);
            }
            out[i] = max_value;
        }
    });
}

    } // kernel
} // xsdnn
