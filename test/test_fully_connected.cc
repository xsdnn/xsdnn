//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../include/utils/grad_checker.h"
using namespace xsdnn;

TEST(fc, forward_1) {
    fully_connected fc(6, 3);
    fc.weight_init(weight_init::constant(0.0f));
    fc.bias_init(weight_init::constant(1.0f));

    mat_t in = {0, 1, 2, 3, 4, 5};
    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];
    mat_t e = {1, 1, 1};

    for (size_t i = 0; i < e.size(); i++) {
#ifdef MM_USE_DOUBLE
        EXPECT_DOUBLE_EQ(o[i], e[i]);
#else
        EXPECT_FLOAT_EQ(o[i], e[i]);
#endif
    }
}

TEST(fc, forward_2) {
    fully_connected fc(6, 3);
    fc.weight_init(weight_init::constant(0.5f));
    fc.bias_init(weight_init::constant(0.3f));

    mat_t in = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];
    mat_t e = {7.8f, 7.8f, 7.8f};

    for (size_t i = 0; i < e.size(); i++) {
#ifdef MM_USE_DOUBLE
        EXPECT_DOUBLE_EQ(o[i], e[i]);
#else
        EXPECT_FLOAT_EQ(o[i], e[i]);
#endif
    }
}

TEST(fc, forward_nobias) {
    fully_connected fc(6, 3, false);
    fc.weight_init(weight_init::constant(0.5f));

    mat_t in = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];
    mat_t e = {7.5f, 7.5f, 7.5f};

    for (size_t i = 0; i < e.size(); i++) {
#ifdef MM_USE_DOUBLE
        EXPECT_DOUBLE_EQ(o[i], e[i]);
#else
        EXPECT_FLOAT_EQ(o[i], e[i]);
#endif
    }
}

TEST(fc, backward) {
    fully_connected fc(50, 100);
    fc.set_parallelize(false);
    GradChecker checker(&fc, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(fc, forward_1_parallel) {
    fully_connected fc(6, 3);
    fc.weight_init(weight_init::constant(0.0f));
    fc.bias_init(weight_init::constant(1.0f));

    mat_t in = {0, 1, 2, 3, 4, 5};
    fc.set_num_threads(std::thread::hardware_concurrency());
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];
    mat_t e = {1, 1, 1};

    for (size_t i = 0; i < e.size(); i++) {
#ifdef MM_USE_DOUBLE
        EXPECT_DOUBLE_EQ(o[i], e[i]);
#else
        EXPECT_FLOAT_EQ(o[i], e[i]);
#endif
    }
}

TEST(fc, forward_2_parallel) {
    fully_connected fc(6, 3);
    fc.weight_init(weight_init::constant(0.5f));
    fc.bias_init(weight_init::constant(0.3f));

    mat_t in = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    fc.set_num_threads(std::thread::hardware_concurrency());
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];
    mat_t e = {7.8f, 7.8f, 7.8f};

    for (size_t i = 0; i < e.size(); i++) {
#ifdef MM_USE_DOUBLE
        EXPECT_DOUBLE_EQ(o[i], e[i]);
#else
        EXPECT_FLOAT_EQ(o[i], e[i]);
#endif
    }
}

TEST(fc, forward_nobias_parallel) {
    fully_connected fc(6, 3, false);
    fc.weight_init(weight_init::constant(0.5f));

    mat_t in = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    fc.set_num_threads(std::thread::hardware_concurrency());
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];
    mat_t e = {7.5f, 7.5f, 7.5f};

    for (size_t i = 0; i < e.size(); i++) {
#ifdef MM_USE_DOUBLE
        EXPECT_DOUBLE_EQ(o[i], e[i]);
#else
        EXPECT_FLOAT_EQ(o[i], e[i]);
#endif
    }
}

TEST(fc, backward_parallel) {
    fully_connected fc(50, 100);
    fc.set_num_threads(std::thread::hardware_concurrency());
    GradChecker checker(&fc, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(fc, cerial) {
    fully_connected fc(50, 100);
    ASSERT_TRUE(utils::cerial_testing(fc));
}