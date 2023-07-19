//
// Created by rozhin on 19.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/input.h>
#include <utils/macro.h>

namespace xsdnn {

std::vector<shape3d> input::in_shape() const {
    return {shape_};
}

std::vector<shape3d> input::out_shape() const {
    return {shape_};
}

std::string input::layer_type() const {
    return "input";
}

size_t input::fan_in_size() const {
    return shape_.size();
}

size_t input::fan_out_size() const {
    return shape_.size();
}

void input::forward_propagation(const std::vector<tensor_t *> &in_data,
                                std::vector<tensor_t *> &out_data) {
    *out_data[0] = *in_data[0];
}

void input::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                             std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
    XS_UNUSED_PARAMETER(in_data);
    XS_UNUSED_PARAMETER(out_data);
    XS_UNUSED_PARAMETER(out_grad);
    XS_UNUSED_PARAMETER(in_grad);
}

} // xsdnn