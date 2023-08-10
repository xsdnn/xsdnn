//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/batch_norm/bn_fwd_xs_impl.h>
#include <core/framework/threading.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace xsdnn {
    namespace kernel {

void compute_mean(const tensor_t& in, size_t channels, size_t spatial_size, mat_t& mean) {
    for (size_t sample = 0; sample < in.size(); ++sample) {
        for (size_t c = 0; c < channels; ++c) {
            mm_scalar& m = mean.at(c);
            const auto start = in[sample].begin() + (c * spatial_size);
            m = std::accumulate(start, start + spatial_size, m);
        }
    }
    std::transform(mean.begin(), mean.end(), mean.begin(),
                   [=](mm_scalar x) { return x / ( in.size() * spatial_size ) ; });
}

void compute_stddev(const tensor_t& in, size_t channels, size_t spatial_size,
                    mat_t& mean, mat_t& stddev, mm_scalar eps) {
    for (size_t i = 0; i < in.size(); i++) {
        for (size_t j = 0; j < channels; j++) {
            mm_scalar &rstddev    = stddev[j];
            const auto it    = in[i].begin() + (j * spatial_size);
            const mm_scalar ex = mean[j];
            rstddev             = std::accumulate(it, it + spatial_size, rstddev,
                                               [ex](mm_scalar current, mm_scalar x) {
                                                   return current + std::pow(x - ex, mm_scalar{2.0f});
                                               });
        }
    }
    std::transform(stddev.begin(), stddev.end(), stddev.begin(),
                   [=](mm_scalar x) { return x / (std::max(float_t{1.0f},
                                                           static_cast<float_t>(in.size() * spatial_size) - float_t{1.0f}));
                   });

    for (size_t i = 0; i < channels; i++) {
        stddev[i] = sqrt(stddev[i] + eps);
    }
}

void compute_moments(const tensor_t& in, shape3d& shape, mat_t& mean, mat_t& stddev, mm_scalar eps) {
    size_t channels = shape.D;
    size_t spatial_size = shape.area();

    compute_mean(in, channels, spatial_size, mean);
    compute_stddev(in, channels, spatial_size, mean, stddev, eps);
}

void batch_normalization_fwd_xs_impl(const tensor_t& in,
                                     const mat_t& gamma,
                                     const mat_t& beta,
                                     tensor_t& out,
                                     params::bnorm& p,
                                     bool parallelize,
                                     size_t nthreads) {
    mat_t& mean = (p.phase_ == op_mode::train) ? p.stat_holder["mean_running_"] : p.stat_holder["mean_"];
    mat_t& stddev = (p.phase_ == op_mode::train) ? p.stat_holder["stddev_running_"] : p.stat_holder["stddev_"];

    if (p.phase_ == op_mode::train) {
        compute_moments(in, p.in_shape_, mean, stddev, p.eps_);
    }

    size_t channel = p.in_shape_.D;
    size_t spatial_size = p.in_shape_.area();

    concurrency::TryParallelFor(parallelize, nthreads, in.size(), [&](size_t sample) {
        const mm_scalar* in_ptr = in[sample].data();
        mm_scalar* out_ptr = out[sample].data();

        for (size_t c = 0; c < channel; ++c) {
            mm_scalar& m = mean[c];
            mm_scalar& s = stddev[c];
            for (size_t k = 0; k < spatial_size; ++k) {
                *out_ptr++ = gamma[c] * (*in_ptr++ - m) / s + beta[c];
            }
        }
    });

}

    } // kernel
} // xsdnn