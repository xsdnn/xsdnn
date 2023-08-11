//
// Created by rozhin on 19.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/input.h>
#include <utils/macro.h>

namespace xsdnn {

std::vector<shape3d> Input::in_shape() const {
    return {shape_};
}

std::vector<shape3d> Input::out_shape() const {
    return {shape_};
}

std::string Input::layer_type() const {
    return "Input";
}

void Input::forward_propagation(const std::vector<tensor_t *> &in_data,
                                std::vector<tensor_t *> &out_data) {
    *out_data[0] = *in_data[0];
}

void Input::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                             std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
    XS_UNUSED_PARAMETER(in_data);
    XS_UNUSED_PARAMETER(out_data);
    XS_UNUSED_PARAMETER(out_grad);
    XS_UNUSED_PARAMETER(in_grad);
}

} // xsdnn