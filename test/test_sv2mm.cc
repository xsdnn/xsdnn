//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
using namespace mmpack;

#define M 1
#define K 9
#define N 6

TEST(sv2mm, _01) {
    xsdnn::mat_t A = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    xsdnn::mat_t B = {9, 10, 11, 12, 13, 14};
    xsdnn::mat_t C(K * N);

    mm_scalar ExpectedArr[] {   0,   0,   0,   0,   0,   0,
                                9,  10,  11,  12,  13,  14,
                                18,  20,  22,  24,  26,  28,
                                27, 30,  33,  36,  39,  42,
                                36,  40,  44,  48,  52,  56,
                                45,  50,  55,  60,  65,  70,
                                54,  60,  66,  72,  78,  84,
                                63,  70,  77,  84,  91,  98,
                                72,  80,  88,  96, 104, 112  };

    MmVectorToMatrixMul(
            K, N,
            1.0f,
            A.data(),
            B.data(),
            0.0f,
            C.data(), N);

    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            ASSERT_EQ(C[i * N + j], ExpectedArr[i * N + j]);
        }
    }
}

