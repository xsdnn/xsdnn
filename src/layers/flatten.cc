//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/flatten.h>
#include <utils/macro.h>

namespace xsdnn {

    std::string flatten::layer_type() const {
        return "flatten";
    }

    void flatten::set_in_shape(const xsdnn::shape3d in_shape) {
        in_shape_ = in_shape;
    }

    std::vector<shape3d> flatten::in_shape() const {
        return { in_shape_ };
    }

    std::vector<shape3d> flatten::out_shape() const {
        return { shape3d(in_shape_.size(), 1, 1) };
    }

    void flatten::forward_propagation(const std::vector<tensor_t *> &in_data,
                                      std::vector<tensor_t *> &out_data) {
        *out_data[0] = *in_data[0];
    }

    void flatten::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                                   std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
        XS_UNUSED_PARAMETER(in_data);
        XS_UNUSED_PARAMETER(out_data);
        XS_UNUSED_PARAMETER(out_grad);
        XS_UNUSED_PARAMETER(in_grad);
    }

} // xsdnn