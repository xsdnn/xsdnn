//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../sgemm.h"

namespace mmpack {

void ReferenceGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
) {
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
        if (TransB == CBLAS_TRANSPOSE::CblasNoTrans) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + (m * lda);
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        b += ldb;
                        a += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        } else {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + (m * lda);
                    const float* b = B + (n * ldb);
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        a += 1;
                        b += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        }
    } else {
        if (TransB == CBLAS_TRANSPOSE::CblasNoTrans) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + m;
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        a += lda;
                        b += ldb;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        } else {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + m;
                    const float* b = B + n * ldb;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        a += lda;
                        b += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        }
    }
}

void
MmGemmOp(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
) {
    ReferenceGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void
MmGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
) {
    MmGemmOp(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
} // mmpack

