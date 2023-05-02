//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_UTIL_H
#define XSDNN_UTIL_H

#include "xs_error.h"

namespace xsdnn {

enum class tensor_type : int32_t {
    // input/output data
    data = 0x0001000,  // input/output data

    // trainable parameters
    weight = 0x0002000,
    bias   = 0x0002001,

    label = 0x0004000,
};

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

}

#endif //XSDNN_UTIL_H
