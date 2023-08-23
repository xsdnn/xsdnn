//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/util.h>
#include <cmath>

namespace xsdnn {

bool is_trainable_concept(tensor_type type_) {
    bool value;
    switch (type_) {
        case tensor_type::data:
            value = false;
            break;
        case tensor_type::label:
            value = false;
            break;
        case tensor_type::weight:
            value = true;
            break;
        case tensor_type::bias:
            value = true;
            break;
        default:
            throw xs_error("Unsupported tensor type");
    }
    return value;
}

// TODO: решить проблему с типами
std::pair<size_t, size_t> find_data_idx(const std::vector<tensor_type>& t1,
                                        const std::vector<tensor_type>& t2) {
    auto data_idx = std::pair<size_t, size_t>(-1, -1);

    for (size_t i = 0; i < t1.size(); ++i) {
        if (t1[i] == tensor_type::data) {
            data_idx.first = i;
        }
    }

    for (size_t i = 0; i < t2.size(); ++i) {
        if (t2[i] == tensor_type::data) {
            data_idx.second = i;
        }
    }

    if (data_idx.first == -1 || data_idx.second == -1) {
        throw xs_error("Not found \'data\' tensor type.");
    }
    return data_idx;
}

std::vector<tensor_type> define_input_bias_condition(bool has_bias) {
    if (has_bias) {
        return {tensor_type::data, tensor_type::weight, tensor_type::bias};
    } else {
        return {tensor_type::data, tensor_type::weight};
    }
}

std::ostream& operator<<(std::ostream& out, tensor_type type) {
    switch (type) {
        case tensor_type::data:
            out << "data";
            break;
        case tensor_type::weight:
            out << "weight";
            break;
        case tensor_type::bias:
            out << "bias";
            break;
    }
    return out;
}

size_t calc_pool_shape(size_t in_size,
                       size_t padding,
                       size_t stride,
                       padding_mode pad_type,
                       bool   ceil) {
    size_t out_size;

    if (pad_type == padding_mode::valid) {
        out_size = in_size - padding + 1;
    } else if (pad_type == padding_mode::same) {
        out_size = in_size;
    } else {
        xs_error("Unsupported padding mode");
    }

    float ir_out_size = (out_size + stride - 1) / stride;
    return (ceil ? std::ceil(ir_out_size) : std::floor(ir_out_size));
}

bool is_1D_tensor(shape3d in) {
    return in.C == 1 && in.H == 1 && in.W > 1;
}

bool is_2D_tensor(shape3d in) {
    return in.C > 1 && in.H > 1 && in.W > 1;
}

} // xsdnn