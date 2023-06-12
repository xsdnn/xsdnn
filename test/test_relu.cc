//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
#include "../include/utils/grad_checker.h"
using namespace xsdnn;

TEST(relu, forward) {
    relu rl(5);
    mat_t in = {-1, -0.5, 0.0, 0.5, 1};
    rl.setup(false);
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
    GradChecker checker(&rl, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}