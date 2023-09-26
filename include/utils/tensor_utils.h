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

void fill(mm_scalar* p_, size_t size, mm_scalar val);
void fill(BTensor& t_, mm_scalar val);

    } // tensorize
} // xsdnn

#endif //XSDNN_TENSOR_UTILS_H
