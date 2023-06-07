//
// Created by rozhin on 07.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../include/framework/threadpool.h"
#include <gtest/gtest.h>
using namespace xsdnn::concurrency;

typedef std::vector<float> vec_t;
typedef std::vector<vec_t> tensor_t;

void sum(vec_t& v, float& s) {
    for (size_t i = 0; i < v.size(); i++) {
        s += v[i];
    }
}

TEST(threadpool, simple_test) {
    tensor_t T = {{1, 2},
                  {8, 9, 10, 0.1},
                  {0, -1, -1, -1, 100},
                  {0.1234, 0.6788},
                  {0, 123, -987},
                  {1},
                  {9, -7, 0.3697},
                  {-1.123}};

    vec_t S = {0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t i = 0; i < T.size(); i++) {
        ThreadPool.add_task(sum,
                    std::ref(T[i]),
                    std::ref(S[i]));
    }
    ThreadPool.wait_all();

    ASSERT_FLOAT_EQ(S[0], 3);
    ASSERT_FLOAT_EQ(S[1], 27.1);
    ASSERT_FLOAT_EQ(S[2], 97);
    ASSERT_FLOAT_EQ(S[3], 0.8022);
    ASSERT_FLOAT_EQ(S[4], -864);
    ASSERT_FLOAT_EQ(S[5], 1);
    ASSERT_FLOAT_EQ(S[6], 2.3697);
    ASSERT_FLOAT_EQ(S[7], -1.123);
}