//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../include/utils/grad_checker.h"
using namespace xsdnn;

// TODO: поменять обратно на chw

TEST(abs, forward) {
    shape3d shape_(3, 224, 224);
    xsdnn::abs abs1(shape_);
    mat_t in_data(shape_.size());
    utils::random_init(in_data.data(), shape_.size());

    abs1.set_in_data({{ in_data }});
    abs1.set_parallelize(false);
    abs1.setup(false);

    abs1.forward();
    mat_t out = abs1.output()[0][0];

    for (size_t h = 0; h < shape_.H; ++h) {
        for (size_t w = 0; w < shape_.W; ++w) {
            for (size_t c = 0; c < shape_.C; ++c) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
                ASSERT_FLOAT_EQ(std::abs(in_data[shape_(c, h, w)]), out[shape_(c, h, w)]);
#endif
            }
        }
    }
}

TEST(abs, backward) {
    shape3d shape_(1, 64, 64);
    xsdnn::abs abs1(shape_);
    abs1.set_parallelize(false);
    GradChecker checker(&abs1, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(abs, backward_parallel) {
    shape3d shape_(1, 64, 64);
    xsdnn::abs abs1(shape_);
    abs1.set_parallelize(true);
    abs1.set_num_threads(std::thread::hardware_concurrency());
    GradChecker checker(&abs1, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(abs, cerial) {
    shape3d shape_(3, 224, 224);
    xsdnn::abs abs1(shape_);
    ASSERT_TRUE(utils::cerial_testing(abs1));
}