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

void Input::forward_propagation(const std::vector<BTensor *> &in_data,
                                std::vector<BTensor *> &out_data) {
    *out_data[0] = *in_data[0];
}

} // xsdnn