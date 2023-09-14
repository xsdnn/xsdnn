//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/hard_sigmoid.h>

namespace xsdnn {

void hard_sigmoid::forward_activation(const xsdnn::mat_t &in_data, xsdnn::mat_t &out_data) {
    shape3d in_shape = this->in_shape()[0];
    std::copy(in_data.begin(), in_data.end(), out_data.begin());
    mmpack::MmActivation(&activationHolder_, out_data.data(), in_shape.H, in_shape.W, in_shape.W);
}

void hard_sigmoid::back_activation(const xsdnn::mat_t &in_data, const xsdnn::mat_t &out_data,
                                   const xsdnn::mat_t &out_grad, xsdnn::mat_t &in_grad) {
    throw xs_error("[hard_sigmoid bwd] NotImplementedYet");
}

std::pair<mm_scalar, mm_scalar> hard_sigmoid::out_value_range() const {
    return std::make_pair(mm_scalar (0.1), mm_scalar (0.9));
}

std::string hard_sigmoid::layer_type() const {
    return "hard_sigmoid";
}

}