//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../include/utils/grad_checker.h"
using namespace xsdnn;

TEST(add, forward) {
    shape3d shape_ = shape3d(28, 28, 3);
    xsdnn::and_layer and_(shape_);
    mat_t in1(shape_.size()), in2(shape_.size());
    utils::value_init(in1.data(), 0.0f, shape_.size());
    utils::value_init(in2.data(), 1.0f, shape_.size());

    and_.set_in_data({{in1}, {in2}});
    and_.set_parallelize(false);
    and_.forward();

    auto out = and_.output()[0][0];
    mat_t expected(shape_.size());
    tensorize::fill(expected.data(), expected.size(), 0.0f);

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

//TEST(add, backward) {
//    // TODO: довести до ума великомученника and layer
//    shape3d shape_ = shape3d(28, 28, 3);
//    xsdnn::and_layer and_(shape_);
//    and_.set_parallelize(false);
//    GradChecker checker(&and_, GradChecker::mode::random);
//    GradChecker::status STATUS = checker.run();
//    ASSERT_EQ(STATUS, GradChecker::status::ok);
//}

TEST(add, cerial) {
    shape3d shape_ = shape3d(28, 28, 3);
    xsdnn::and_layer and_(shape_);
    ASSERT_TRUE(utils::cerial_testing(and_));
}
