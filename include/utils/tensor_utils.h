//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_UTILS_H
#define XSDNN_TENSOR_UTILS_H

#include "tensor.h"
#include "../mmpack/mmpack.h"
using namespace mmpack;

namespace xsdnn {
    namespace tensorize {

void fill(xsDtype dtype, mat_t* p_, size_t size, mm_scalar val);
void fill(tensor_t& t_, mm_scalar val);

    } // tensorize
} // xsdnn

#endif //XSDNN_TENSOR_UTILS_H
