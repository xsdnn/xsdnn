//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/weight_init.h>
#include <utils/macro.h>

namespace xsdnn {
    namespace weight_init {

void xavier::fill(tensor_t* data, size_t fan_in, size_t fan_out) {
//    const mm_scalar bias = std::sqrt(scale_ / (fan_in + fan_out));
//    uniform_rand(data, data->shape().size(), -bias, bias);
}

void constant::fill(tensor_t* data, size_t fan_in, size_t fan_out) {
//    XS_UNUSED_PARAMETER(fan_in);
//    XS_UNUSED_PARAMETER(fan_out);
//    tensorize::fill(data, size, scale_);
}

} // weight_init
} // xsdnn