//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/mul.h>

namespace xsdnn {

    std::vector<shape3d> mul::in_shape() const {
        return {in_shape_, in_shape_};
    }

    std::vector<shape3d> mul::out_shape() const {
        return { in_shape_ };
    }

    std::string mul::layer_type() const {
        return "mul";
    }

    void mul::forward_propagation(const std::vector<tensor_t *> &in_data,
                                  std::vector<tensor_t *> &out_data) {
        throw xs_error("[mul fwd] Not Implemented Yet");
    }

} // xsdnn