//
// Created by rozhin on 31.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>

TEST(muladd, simple_check) {
    float A[] = {0, 1, 2, 3, 4};
    float B[] = {0, 1, 2, 3, 4};
    float C[] = {0, 0, 0, 0, 0};



    float expected[] = {0, 1, 4, 9, 16};
    mmpack::MmMulAdd(&A[0], &B[0], &C[0], 5);

    ASSERT_EQ(expected[0], C[0]);
    ASSERT_EQ(expected[1], C[1]);
    ASSERT_EQ(expected[2], C[2]);
    ASSERT_EQ(expected[3], C[3]);
    ASSERT_EQ(expected[4], C[4]);
}

TEST(muladd, with_negative_samples) {
    float A[] = {0, -1, 2, -3, 4};
    float B[] = {0, 1, -2, 3, -4};
    float C[] = {0, 0, 0, 0, 0};



    float expected[] = {0, -1, -4, -9, -16};
    mmpack::MmMulAdd(&A[0], &B[0], &C[0], 5);

    ASSERT_EQ(expected[0], C[0]);
    ASSERT_EQ(expected[1], C[1]);
    ASSERT_EQ(expected[2], C[2]);
    ASSERT_EQ(expected[3], C[3]);
    ASSERT_EQ(expected[4], C[4]);
}

TEST(muladd, with_floating_point) {
    float A[] = {0, -1.5, 2, -3.5, 4};
    float B[] = {0, 1, -2.5, 3, -4.5};
    float C[] = {0, 0, 0, 0, 0};



    float expected[] = {0, -1.5, -5.0, -10.5, -18.0};
    mmpack::MmMulAdd(&A[0], &B[0], &C[0], 5);

#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
    ASSERT_FLOAT_EQ(expected[0], C[0]);
    ASSERT_FLOAT_EQ(expected[1], C[1]);
    ASSERT_FLOAT_EQ(expected[2], C[2]);
    ASSERT_FLOAT_EQ(expected[3], C[3]);
    ASSERT_FLOAT_EQ(expected[4], C[4]);
#endif
}

void
ReferenceMulAdd(
        const float* A,
        const float* B,
        float* C,
        size_t size
)
{
    for (size_t i = 0; i < size; ++i) {
        *C += *A * *B;

        A += 1;
        B += 1;
        C += 1;
    }
}

TEST(dot, stress) {
    xsdnn::default_random_engine rng(42);
    std::vector<mm_scalar> A(1000000);
    std::vector<mm_scalar> B(1000000);
    std::vector<mm_scalar> C(1000000);
    std::vector<mm_scalar> C_ref(1000000);

    for (size_t K = 0; K < 10; K++) {

        for (size_t i = 0; i < A.size(); i++) {
            A[i] = rng.rand();
            B[i] = rng.rand();
            C[i] = 0.0f;
            C_ref[i] = 0.0f;
        }

        mmpack::MmMulAdd(A.data(), B.data(), C.data(), A.size());
        ReferenceMulAdd(A.data(), B.data(), C_ref.data(), A.size());

        for (size_t j = 0; j < A.size(); ++j) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
            ASSERT_FLOAT_EQ(C_ref[j], C[j]);
#endif
        }
    }
}
