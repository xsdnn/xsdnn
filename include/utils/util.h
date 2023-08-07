//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_UTIL_H
#define XSDNN_UTIL_H

#include "xs_error.h"
#include <vector>

namespace xsdnn {

enum class tensor_type : int32_t {
    // input/output data
    data = 0x0001000,  // input/output data

    // trainable parameters
    weight = 0x0002000,
    bias   = 0x0002001,

    label = 0x0004000,
};

bool is_trainable_concept(tensor_type type_);

std::pair<size_t, size_t> find_data_idx(const std::vector<tensor_type>& t1,
                                        const std::vector<tensor_type>& t2);


std::vector<tensor_type> define_input_bias_condition(bool has_bias);

enum class op_mode {
    inference = 0,
    train = 1
};

}

#endif //XSDNN_UTIL_H
