//
// Created by rozhin on 09.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/output.h>
#include <utils/macro.h>

namespace xsdnn {

void Output::set_in_shape(const xsdnn::shape3d in_shape) {
    shape_ = in_shape;
}

std::vector<shape3d> Output::in_shape() const {
    return {shape_};
}

std::vector<shape3d> Output::out_shape() const {
    return {shape_};
}

std::string Output::layer_type() const {
    return "Output";
}

void Output::forward_propagation(const std::vector<BTensor*> &in_data,
                                std::vector<BTensor*> &out_data) {
    *out_data[0] = *in_data[0];
}

} // xsdnn