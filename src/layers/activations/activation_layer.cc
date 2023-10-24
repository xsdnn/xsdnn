//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/activation_layer.h>

namespace xsdnn {

activation_layer::activation_layer()
    :
    layer({tensor_type::data}, {tensor_type::data}, xsDtype::kXsFloat32),
    in_shape_(0, 0, 0), initialized_(false) { init_backend(); }

activation_layer::activation_layer(const size_t in_size)
    :
    layer({tensor_type::data}, {tensor_type::data}, xsDtype::kXsFloat32),
    in_shape_(1, 1, in_size), initialized_(false) { init_backend(); }

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

void activation_layer::init_backend() {
    fwd_kernel_.reset(new core::ActivationFwdKernel);
}

// TODO: сделать проход вперед через ctx, для передачи типа данных
void
activation_layer::forward_propagation(const std::vector<tensor_t *> &in_data,
                                      std::vector<tensor_t *> &out_data) {
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.set_engine(layer::engine());               // TODO: сделать поддержку разных движков на активации
    fwd_ctx_.set_parallelize(layer::parallelize());
    fwd_ctx_.set_num_threads(layer::num_threads_);
    fwd_ctx_.set_dtype(layer::dtype());

    if (!initialized_) init_params();
    fwd_kernel_->Compute(fwd_ctx_, p_);
}

} // xsdnn