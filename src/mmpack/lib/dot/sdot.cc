//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../dot.h"

namespace mmpack {

float
ReferenceDot(
        const float* A,
        const float* B,
        size_t size
) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        const float* a = A + i;
        const float* b = B + i;
        sum += (*a * *b);
    }
    return sum;
}

float
MmDotOp(
        const float* A,
        const float* B,
        size_t size
) {
    return ReferenceDot(A, B, size);
}

float
MmDot(
        const float* A,
        const float* B,
        size_t size
) {
    return MmDotOp(A, B, size);
}

} // mmpack