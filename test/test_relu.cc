//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "../include/utils/grad_checker.h"
#include "test_utils.h"
using namespace xsdnn;

TEST(relu, forward) {
    relu rl(5);
    mat_t in = {-1, -0.5, 0.0, 0.5, 1};
    rl.setup(false);
    rl.set_parallelize(false);
    rl.set_in_data({{in}});
    rl.forward();
    mat_t out = rl.output()[0][0];
    mat_t e = {0.0f, 0.0f, 0.0f, 0.5, 1.0f};
    for (size_t i = 0; i < 5; i++) {
        ASSERT_EQ(out[i], e[i]);
    }
}

TEST(relu, backward) {
    relu rl(784);
    rl.set_parallelize(false);
    GradChecker checker(&rl, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(relu, forward_paralell) {
    relu rl(5);
    mat_t in = {-1, -0.5, 0.0, 0.5, 1};
    rl.setup(false);
    rl.set_num_threads(std::thread::hardware_concurrency());
    rl.set_in_data({{in}});
    rl.forward();
    mat_t out = rl.output()[0][0];
    mat_t e = {0.0f, 0.0f, 0.0f, 0.5, 1.0f};
    for (size_t i = 0; i < 5; i++) {
        ASSERT_EQ(out[i], e[i]);
    }
}

TEST(relu, backward_parallel) {
    relu rl(784);
    rl.set_num_threads(std::thread::hardware_concurrency());
    GradChecker checker(&rl, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(relu, cerial) {
    relu rl(784);
    ASSERT_TRUE(utils::cerial_testing(rl));
}