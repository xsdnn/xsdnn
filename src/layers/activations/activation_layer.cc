//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/activation_layer.h>
#include <core/framework/threading.h>

namespace xsdnn {

activation_layer::activation_layer()
    :
    layer({TypeHolder(tensor_type::data, XsDtype::F32)}, {TypeHolder(tensor_type::data, XsDtype::F32)}),
    in_shape_(0, 0, 0) {}

activation_layer::activation_layer(const size_t in_size)
    :
    layer({TypeHolder(tensor_type::data, XsDtype::F32)}, {TypeHolder(tensor_type::data, XsDtype::F32)}),
    in_shape_(1, 1, in_size) {}

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
activation_layer::forward_propagation(const std::vector<BTensor*> &in_data,
                                      std::vector<BTensor*> &out_data) {
    const BTensor & in_ = *in_data[0];
    BTensor& out_ = *out_data[0];

    concurrency::TryParallelFor(layer::parallelize_,
                                layer::num_threads_,
                                in_.size(),
                                [&](size_t i) {
            forward_activation(in_[i], out_[i]);
    });
}

} // xsdnn