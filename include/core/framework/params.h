//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_PARAMS_H
#define XSDNN_PARAMS_H

#include <cstddef>
#include <unordered_map>
#include "../../utils/tensor_shape.h"
#include "../../utils/util.h"

namespace xsdnn {
    namespace params {

struct fully {
    size_t in_size_;
    size_t out_size_;
    bool   has_bias_;
};

struct bnorm {
    shape3d in_shape_;
    mm_scalar momentum_;
    mm_scalar eps_;
    op_mode phase_;
    std::unordered_map<std::string, mat_t> param_holder;

    bool statistic_initialized {false};
};

    } // params
} // xsdnn

#endif //XSDNN_PARAMS_H
