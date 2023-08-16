//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <mmpack/v2mm.h>

namespace mmpack {

void
ReferenceVectorToMatrixMul(
        size_t K,
        size_t N,
        float alpha,
        const float* A,
        const float* B,
        float beta,
        float* C,
        size_t ldc
) {
    for (size_t k = 0; k < K; k++) {
        const float* a = A + k;
        float* c = C + (k * ldc);
        for (size_t n = 0; n < N; n++) {
            const float* b = B + n;
            *c = (*c * beta) + (*a * *b * alpha);
            c += 1;
        }
    }
}

void
MmVectorToMatrixMulOp(
        size_t K,
        size_t N,
        float alpha,
        const float* A,
        const float* B,
        float beta,
        float* C,
        size_t ldc
) {
    ReferenceVectorToMatrixMul(K, N, alpha, A, B, beta, C, ldc);
}

void
MmVectorToMatrixMul(
        size_t K,
        size_t N,
        float alpha,
        const float* A,
        const float* B,
        float beta,
        float* C,
        size_t ldc
) {
    MmVectorToMatrixMulOp(K, N, alpha, A, B, beta, C, ldc);
}

} // mmpack
