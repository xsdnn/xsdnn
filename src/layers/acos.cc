//
// Created by rozhin on 27.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/acos.h>
#include <utils/macro.h>
#include <core/framework/threading.h>

namespace xsdnn {

    void acos::set_in_shape(const xsdnn::shape3d in_shape) {
        shape_ = in_shape;
    }

    std::vector<shape3d> acos::in_shape() const {
        return {shape_};
    }

    std::vector<shape3d> acos::out_shape() const {
        return {shape_};
    }

    std::string acos::layer_type() const {
        return "acos";
    }

    void acos::forward_propagation(const std::vector<BTensor *> &in_data,
                                  std::vector<BTensor *> &out_data) {
        const BTensor& in = *in_data[0];
        BTensor& out = *out_data[0];

        concurrency::TryParallelFor(this->parallelize_, this->num_threads_, in.size(), [&](size_t sample) {
            gsl::span<const float> in_sample = in[sample].GetDataAsSpan<float>();
            gsl::span<float> out_sample = out[sample].GetMutableDataAsSpan<float>();
            for (size_t j = 0; j < in_sample.size(); ++j) {
                out_sample[j] = std::acos(in_sample[j]);
            }
        });
    }

} // xsdnn