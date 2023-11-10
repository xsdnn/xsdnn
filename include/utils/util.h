//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_UTIL_H
#define XSDNN_UTIL_H

#include "xs_error.h"
#include "tensor_shape.h"
#include <vector>
#include <iostream>

namespace xsdnn {

enum class tensor_type : int32_t {
    // input/output data
    data = 0x0001000,  // input/output data

    // trainable parameters
    weight = 0x0002000,
    bias   = 0x0002001,

    label = 0x0004000,
};

std::ostream& operator<<(std::ostream& out, tensor_type type);

bool is_trainable_concept(tensor_type type_);

std::pair<size_t, size_t> find_data_idx(const std::vector<tensor_type>& t1,
                                        const std::vector<tensor_type>& t2);


std::vector<tensor_type> define_input_bias_condition(bool has_bias);

enum class op_mode {
    inference = 0,
    train = 1
};

enum class padding_mode {
    same = 0,
    same_lower = 1,
    same_upper = 2,
    valid = 3,
    notset = 4
};

size_t calc_pool_shape(size_t in_size,
                       size_t padding,
                       size_t stride,
                       padding_mode pad_type,
                       bool   ceil);

size_t calc_conv_padding_shape();

bool is_1D_tensor(shape3d in);
bool is_2D_tensor(shape3d in);

std::string convert_pad_to_string(padding_mode mode);
padding_mode convert_string_to_pad(std::string mode);



}

#endif //XSDNN_UTIL_H
