//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/activation_layer.h>

namespace xsdnn {

activation_layer::activation_layer()
    :
    layer({tensor_type::data}, {tensor_type::data}),
    in_shape_(0, 0, 0) {}

activation_layer::activation_layer(const size_t in_size)
    :
    layer({tensor_type::data}, {tensor_type::data}),
    in_shape_(in_size, 1, 1) {}

activation_layer::activation_layer(const xsdnn::activation_layer &) = default;
activation_layer &activation_layer::operator=(const xsdnn::activation_layer &) = default;

std::vector<shape3d> activation_layer::in_shape() const {
    return {in_shape_};
}

std::vector<shape3d> activation_layer::out_shape() const {
    return {in_shape_};
}

size_t activation_layer::fan_in_size() const {
    throw xs_warning("[Deprecated] Don't use this!");
}

size_t activation_layer::fan_out_size() const {
    throw xs_warning("[Deprecated] Don't use this!");
}

void activation_layer::set_in_shape(const xsdnn::shape3d in_shape) {
    in_shape_ = in_shape;
}

void
activation_layer::forward_propagation(const std::vector<tensor_t *> &in_data,
                                      std::vector<tensor_t *> &out_data) {
    // FIXME: А если data concept не лежит первым в векторе?
    const tensor_t& in_ = *in_data[0];
    tensor_t& out_ = *out_data[0];

    // TODO: parallelize
    for (size_t i = 0; i < in_.size(); ++i) {
        forward_activation(in_[i], out_[i]);
    }
}

void
activation_layer::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                                   std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
    const tensor_t& in_data_ = *in_data[0];
    const tensor_t& out_data_ = *out_data[0];
    const tensor_t& out_grad_ = *out_grad[0];
    tensor_t& in_grad_ = *in_grad[0];

    // TODO: parallelize
    for (size_t i = 0; i < in_data_.size(); ++i) {
        back_activation(in_data_[i], out_data_[i], out_grad_[i], in_grad_[i]);
    }
}

} // xsdnn