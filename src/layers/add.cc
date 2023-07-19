//
// Created by rozhin on 19.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/add.h>
#include <algorithm>
#include <utils/macro.h>

namespace xsdnn {

std::vector<shape3d> add::in_shape() const {
    return std::vector<shape3d>(n_input_, shape_);
}

std::vector<shape3d> add::out_shape() const {
    return {shape_};
}

std::string add::layer_type() const {
    return "add";
}

void add::forward_propagation(const std::vector<tensor_t *> &in_data,
                              std::vector<tensor_t *> &out_data) {
    const tensor_t& in = *in_data[0];
    tensor_t& out = *out_data[0];
    out = in;

    for (size_t sample = 0; sample < in.size(); ++sample) {
        for (size_t i = 1; i < n_input_; i++) {
            std::transform((*in_data[i])[sample].begin(),
                           (*in_data[i])[sample].end(),
                           out[sample].begin(),
                           out[sample].begin(),
                           [](float_t x, float_t y){ return x + y; });
        }
    }
}

void add::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                           std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
    XS_UNUSED_PARAMETER(in_data);
    XS_UNUSED_PARAMETER(out_data);
    for (size_t i = 0; i < n_input_; i++)
        *in_grad[i] = *out_grad[0];
}



} // xsdnn