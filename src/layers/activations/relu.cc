//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/relu.h>

namespace xsdnn {

void relu::forward_activation(const mat_t &in_data, mat_t& out_data) {
//    for (size_t j = 0; j < in_data.size(); ++j) {
//        out_data[j] = std::max(in_data[j], (mm_scalar) 0.0f);
//    }
    shape3d in_shape = this->in_shape()[0];
    mmpack::MmActivationHolder Holder;
    Holder.ActivationType = mmpack::Relu;

    std::copy(in_data.begin(), in_data.end(), out_data.begin());

    mmpack::MmActivation(&Holder, out_data.data(), in_shape.H, in_shape.W, in_shape.W);
}

void relu::back_activation(const mat_t &in_data,
                           const mat_t &out_data,
                           const mat_t &out_grad,
                           mat_t &in_grad) {
    for (size_t j = 0; j < in_data.size(); ++j) {
        in_grad[j] = out_grad[j] * (out_data[j] > mm_scalar (0) ? mm_scalar(1) : mm_scalar(0));
    }
}

std::pair<mm_scalar, mm_scalar> relu::out_value_range() const {
    return {(mm_scalar) 0.1f, (mm_scalar) 0.9f};
}

std::string relu::layer_type() const {
    return "relu";
}

} // xsdnn