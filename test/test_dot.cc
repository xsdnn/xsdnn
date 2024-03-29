//
// Created by rozhin on 03.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>

TEST(dot, simple_check) {
    float A[] = {0, 1, 2, 3, 4};
    float B[] = {0, 1, 2, 3, 4};

    float expected = 30;
    ASSERT_EQ(expected, mmpack::MmDot(&A[0], &B[0], 5));
}

TEST(dot, with_negative_sample) {
    float A[] = {0, -1, -2, -3, -4};
    float B[] = {0, 1, 2, 3, 4};

    float expected = -30;
    ASSERT_EQ(expected, mmpack::MmDot(&A[0], &B[0], 5));
}

TEST(dot, with_floating_point) {
    float A[] = {0, -1.1, -2.2, -3.3, -4.4};
    float B[] = {0.0007, 19873, 2.9147, 3.09751, 4.943169971};

#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
    ASSERT_FLOAT_EQ(mmpack::MmDot(&A[0], &B[0], 5), -21898.686);
#endif
}

TEST(dot, stress) {
    xsdnn::default_random_engine rng(42);
    std::vector<mm_scalar> A(1000000);
    std::vector<mm_scalar> B(1000000);


    for (size_t K = 0; K < 10; K++) {

        for (size_t i = 0; i < A.size(); i++) {
            A[i] = rng.rand();
            B[i] = rng.rand();
        }

        mmpack::MmDot(A.data(), B.data(), 1000);
    }
}
