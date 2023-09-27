//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_UTILS_H
#define XSDNN_TENSOR_UTILS_H

#include "../core/framework/tensor.h"
#include "../mmpack/mmpack.h"
using namespace mmpack;

namespace xsdnn {
    namespace tensorize {

template<typename T>
void fill(tensor_t* data, T value) {
    if (data->dtype() == XsDtype::F32) {
        gsl::span<float> TensorSpan = data->template GetMutableDataAsSpan<float>();
        for (size_t i = 0; i < TensorSpan.size(); ++i) {
            TensorSpan[i] = value;
        }
    } else {
        throw xs_error("[tensorize fill] Unsupported tensor dtype");
    }
}

    } // tensorize
} // xsdnn

#endif //XSDNN_TENSOR_UTILS_H
