//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../include/utils/grad_checker.h"
using namespace xsdnn;

TEST(acos, forward) {
    shape3d shape_(224, 224, 3);
    xsdnn::acos acos1(shape_);
    mat_t in_data(shape_.size());
    utils::random_init(in_data.data(), shape_.size());

    acos1.set_in_data({{ in_data }});
    acos1.set_parallelize(false);
    acos1.setup(false);

    acos1.forward();
    mat_t out = acos1.output()[0][0];

    for (size_t h = 0; h < shape_.H; ++h) {
        for (size_t w = 0; w < shape_.W; ++w) {
            for (size_t c = 0; c < shape_.C; ++c) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
                ASSERT_FLOAT_EQ(std::acos(in_data[shape_(h, w, c)]), out[shape_(h, w, c)]);
#endif
            }
        }
    }
}

TEST(acos, backward) {
    shape3d shape_(64, 64, 1);
    xsdnn::acos acos1(shape_);
    acos1.set_parallelize(false);
    GradChecker checker(&acos1, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(acos, backward_parallel) {
    shape3d shape_(64, 64, 1);
    xsdnn::acos acos1(shape_);
    acos1.set_parallelize(true);
    acos1.set_num_threads(std::thread::hardware_concurrency());
    GradChecker checker(&acos1, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}
TEST(acos, cerial) {
    shape3d shape_(64, 64, 3);
    xsdnn::acos acos1(shape_);
    utils::cerial_testing(acos1);
}

