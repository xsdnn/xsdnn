//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(input, forward) {
    shape3d shape_ = shape3d(3, 224, 224);
    Input in(shape_);
    mat_t in_data(shape_.size());
    utils::random_init(in_data.data(), shape_.size());

    in.set_in_data({{ in_data }});
    in.set_parallelize(false);
    in.setup(false);

    in.forward();
    mat_t out = in.output()[0][0];

    for (size_t h = 0; h < shape_.H; ++h) {
        for (size_t w = 0; w < shape_.W; ++w) {
            for (size_t c = 0; c < shape_.C; ++c) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
                ASSERT_FLOAT_EQ(in_data[shape_(c, h, w)], out[shape_(c, h, w)]);
#endif
            }
        }
    }
}

TEST(input, cerial) {
    shape3d shape_ = shape3d(3, 224, 224);
    Input in(shape_);
    ASSERT_TRUE(utils::cerial_testing(in));
}
