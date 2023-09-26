//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;



TEST(abs, forward) {
    shape3d shape_(3, 224, 224);
    xsdnn::abs abs1(shape_);
    tensor_t in_data(XsDtype::F32, shape_.size(), nullptr);
    utils::random_init(in_data.GetMutableData<float>(), shape_.size());

    abs1.set_in_data({{ in_data }});
    abs1.set_parallelize(false);
    abs1.setup(false);

    abs1.forward();
    tensor_t out = abs1.output()[0][0];

    gsl::span<const float> InSpan = in_data.GetDataAsSpan<float>();
    gsl::span<const float> OutSpan = out.GetDataAsSpan<float>();

    for (size_t h = 0; h < shape_.H; ++h) {
        for (size_t w = 0; w < shape_.W; ++w) {
            for (size_t c = 0; c < shape_.C; ++c) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
                ASSERT_FLOAT_EQ(std::abs(InSpan[shape_(c, h, w)]), OutSpan[shape_(c, h, w)]);
#endif
            }
        }
    }
}

TEST(abs, cerial) {
    shape3d shape_(3, 224, 224);
    xsdnn::abs abs1(shape_);
    ASSERT_TRUE(utils::cerial_testing(abs1));
}