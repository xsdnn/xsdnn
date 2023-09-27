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

void compute_mean(const BTensor & in, size_t channels, size_t spatial_size, tensor_t& mean) {
    gsl::span<float> MeanSpan = mean.GetMutableDataAsSpan<float>();
    for (size_t sample = 0; sample < in.size(); ++sample) {
        gsl::span<const float> InSampleSpan = in[sample].GetDataAsSpan<float>();
        for (size_t c = 0; c < channels; ++c) {
            float& m = MeanSpan[c];
            if (m != 0) m = 0;
            const auto start = InSampleSpan.begin() + (c * spatial_size);
            m = std::accumulate(start, start + spatial_size, m);
        }
    }
    std::transform(MeanSpan.begin(), MeanSpan.end(), MeanSpan.begin(),
                   [=](float x) { return x / ( in.size() * spatial_size ) ; });
}

void compute_stddev(const BTensor& in, size_t channels, size_t spatial_size,
                    tensor_t& mean, tensor_t& stddev, float eps) {
    gsl::span<float> MeanSpan = mean.GetMutableDataAsSpan<float>();
    gsl::span<float> StddevSpan = stddev.GetMutableDataAsSpan<float>();
    for (size_t i = 0; i < in.size(); i++) {
        gsl::span<const float> InSampleSpan = in[i].GetDataAsSpan<float>();
        for (size_t j = 0; j < channels; j++) {
            float& rstddev    = StddevSpan[j];
            if (rstddev != 0) rstddev = 0;
            const auto it    = InSampleSpan.begin() + (j * spatial_size);
            const float ex = MeanSpan[j];
            rstddev             = std::accumulate(it, it + spatial_size, rstddev,
                                               [ex](float current, float x) {
                                                   return current + std::pow(x - ex, float {2.0f});
                                               });
        }
    }
    std::transform(StddevSpan.begin(), StddevSpan.end(), StddevSpan.begin(),
                   [=](float x) { return x / (std::max(float_t{1.0f},
                                                           static_cast<float_t>(in.size() * spatial_size) - float_t{1.0f}));
                   });

    for (size_t i = 0; i < channels; i++) {
        StddevSpan[i] = sqrt(StddevSpan[i] + eps);
    }
}

void compute_moments(const BTensor& in, shape3d& shape, tensor_t& mean, tensor_t& stddev, float eps) {
    size_t channels = shape.C;
    size_t spatial_size = shape.area();

    compute_mean(in, channels, spatial_size, mean);
    compute_stddev(in, channels, spatial_size, mean, stddev, eps);
}

void batch_normalization_fwd_xs_impl(const BTensor& in,
                                     const tensor_t& gamma,
                                     const tensor_t& beta,
                                     BTensor& out,
                                     params::bnorm& p,
                                     bool parallelize,
                                     size_t nthreads) {
    tensor_t& mean = (p.phase_ == op_mode::train) ? p.stat_holder["mean_running_"] : p.stat_holder["mean_"];
    tensor_t& stddev = (p.phase_ == op_mode::train) ? p.stat_holder["stddev_running_"] : p.stat_holder["stddev_"];

    if (p.phase_ == op_mode::train) {
        compute_moments(in, p.in_shape_, mean, stddev, p.eps_);
    }

    size_t channel = p.in_shape_.C;
    size_t spatial_size = p.in_shape_.area();

    gsl::span<const float> MeanSpan = mean.GetDataAsSpan<float>();
    gsl::span<const float> StddevSpan = stddev.GetDataAsSpan<float>();
    gsl::span<const float> GammaSpan = gamma.GetDataAsSpan<float>();
    gsl::span<const float> BetaSpan = beta.GetDataAsSpan<float>();


    concurrency::TryParallelFor(parallelize, nthreads, in.size(), [&](size_t sample) {
        const float* in_ptr = in[sample].GetData<float>();
        float* out_ptr = out[sample].GetMutableData<float>();

        for (size_t c = 0; c < channel; ++c) {
            const float& m = MeanSpan[c];
            const float& s = StddevSpan[c];
            for (size_t k = 0; k < spatial_size; ++k) {
                *out_ptr++ = GammaSpan[c] * (*in_ptr++ - m) / s + BetaSpan[c];
            }
        }
    });

}

    } // kernel
} // xsdnn