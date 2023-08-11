//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../include/utils/grad_checker.h"
using namespace xsdnn;

mat_t make_expected(const tensor_t& in_data) {
    mat_t out;
    out.resize(in_data[0].size());
    tensorize::fill(out.data(), out.size(), 0.0f);

    for (size_t i = 0; i < in_data.size(); ++i) {
        for (size_t j = 0; j < in_data[i].size(); ++j) {
            out[j] += in_data[i][j];
        }
    }
    return out;
}

TEST(add, forward) {
    shape3d shape_ = shape3d(28, 28, 3);
    xsdnn::add add1(4, shape_);
    mat_t in1(shape_.size()), in2(shape_.size()), in3(shape_.size()), in4(shape_.size());
    utils::random_init(in1.data(), shape_.size());
    utils::random_init(in2.data(), shape_.size());
    utils::random_init(in3.data(), shape_.size());
    utils::random_init(in4.data(), shape_.size());

    add1.set_in_data({{in1}, {in2}, {in3}, {in4}});
    add1.forward();
    auto out = add1.output()[0][0];
    auto expected = make_expected({in1, in2, in3, in4});

    for (size_t h = 0; h < shape_.H; ++h) {
        for (size_t w = 0; w < shape_.W; ++w) {
            for (size_t c = 0; c < shape_.C; ++c) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
                ASSERT_FLOAT_EQ(out[shape_(h, w, c)], expected[shape_(h, w, c)]);
#endif
            }
        }
    }
}

TEST(add, backward) {
    shape3d shape_(64, 64, 1);
    xsdnn::add add1(shape_);
    add1.set_parallelize(false);
    GradChecker checker(&add1, GradChecker::mode::random);
    GradChecker::status STATUS = checker.run();
    ASSERT_EQ(STATUS, GradChecker::status::ok);
}

TEST(add, cerial) {
    shape3d shape_(64, 64, 1);
    xsdnn::add add1(shape_);
    ASSERT_TRUE(utils::cerial_testing(add1));
}