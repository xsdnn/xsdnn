//
// Created by rozhin on 09.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/output.h>
#include <utils/macro.h>

namespace xsdnn {

void output::set_in_shape(const xsdnn::shape3d in_shape) {
    shape_ = in_shape;
}

std::vector<shape3d> output::in_shape() const {
    return {shape_};
}

std::vector<shape3d> output::out_shape() const {
    return {shape_};
}

std::string output::layer_type() const {
    return "output";
}

void output::forward_propagation(const std::vector<tensor_t *> &in_data,
                                std::vector<tensor_t *> &out_data) {
    *out_data[0] = *in_data[0];
}

void output::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                             std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
    XS_UNUSED_PARAMETER(in_data);
    XS_UNUSED_PARAMETER(out_data);
    XS_UNUSED_PARAMETER(out_grad);
    XS_UNUSED_PARAMETER(in_grad);
}

} // xsdnn