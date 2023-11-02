//
// Created by rozhin on 02.11.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TRANSPOSE_H
#define XSDNN_TRANSPOSE_H

#include "tensor.h"
#include "tensor_shape.h"

namespace xsdnn {

void xs_single_axis_transpose(mat_t* X, std::vector<size_t> XShape, mat_t* Y, xsDtype dtype, size_t from, size_t to);

} // xsdnn

#endif //XSDNN_TRANSPOSE_H
