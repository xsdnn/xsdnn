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
    std::unordered_map<std::string, mat_t> stat_holder;

    bool statistic_initialized {false};
};

struct max_pool {
    shape3d in_shape_;
    shape3d out_shape_;
    size_t kernel_x_;
    size_t kernel_y_;
    size_t stride_x_;
    size_t stride_y_;
    padding_mode pad_type_;

    std::vector<std::vector<size_t>> out2in;
    std::vector<size_t> in2out;
};

struct global_avg_pool {
    shape3d in_shape_;
    shape3d out_shape_;
};

    } // params
} // xsdnn

#endif //XSDNN_PARAMS_H
