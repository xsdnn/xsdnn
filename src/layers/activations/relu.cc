//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/relu.h>

namespace xsdnn {

void relu::forward_activation(const tensor_t& in_data, tensor_t& out_data) {
    shape3d in_shape = this->in_shape()[0];
    mmpack::MmActivationHolder Holder;
    Holder.ActivationType = mmpack::Relu;

    gsl::span<const float> X = in_data.GetDataAsSpan<float>();
    gsl::span<float> Y = out_data.GetMutableDataAsSpan<float>();

    std::copy(X.begin(), X.end(), Y.begin());

    mmpack::MmActivation(&Holder, Y.data(), in_shape.H, in_shape.W, in_shape.W);
}

std::pair<mm_scalar, mm_scalar> relu::out_value_range() const {
    return {(mm_scalar) 0.1f, (mm_scalar) 0.9f};
}

std::string relu::layer_type() const {
    return "relu";
}

} // xsdnn