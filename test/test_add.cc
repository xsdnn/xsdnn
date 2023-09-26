//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

tensor_t make_expected(const BTensor & in_data) {
    tensor_t out(in_data[0].dtype(), in_data[0].shape().size(), nullptr);
    tensorize::fill(out.GetMutableData<float>(), out.shape().size(), 0.0f);

    gsl::span<float> OutSpan = out.GetMutableDataAsSpan<float>();

    for (size_t i = 0; i < in_data.size(); ++i) {
        gsl::span<const float> InSpan = in_data[i].GetDataAsSpan<float>();
        for (size_t j = 0; j < InSpan.size(); ++j) {
            OutSpan[j] += InSpan[j];
        }
    }
    return out;
}

TEST(add, forward) {
    shape3d shape_ = shape3d(3, 28, 28);
    xsdnn::add add1(4, shape_);
    tensor_t in1(XsDtype::F32, shape_.size(), nullptr),
            in2(XsDtype::F32, shape_.size(), nullptr),
            in3(XsDtype::F32, shape_.size(), nullptr),
            in4(XsDtype::F32, shape_.size(), nullptr);

    utils::random_init(in1.GetMutableData<float>(), shape_.size());
    utils::random_init(in2.GetMutableData<float>(), shape_.size());
    utils::random_init(in3.GetMutableData<float>(), shape_.size());
    utils::random_init(in4.GetMutableData<float>(), shape_.size());

    add1.set_in_data({{in1}, {in2}, {in3}, {in4}});
    add1.forward();
    auto out = add1.output()[0][0];
    auto expected = make_expected({in1, in2, in3, in4});

    gsl::span<const float> ExpectedSpan = expected.GetDataAsSpan<float>();
    gsl::span<const float> OutSpan = out.GetDataAsSpan<float>();

    for (size_t h = 0; h < shape_.H; ++h) {
        for (size_t w = 0; w < shape_.W; ++w) {
            for (size_t c = 0; c < shape_.C; ++c) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
                ASSERT_FLOAT_EQ(OutSpan[shape_(c, h, w)], ExpectedSpan[shape_(c, h, w)]);
#endif
            }
        }
    }
}

TEST(add, cerial) {
    shape3d shape_(1, 64, 64);
    xsdnn::add add1(shape_);
    ASSERT_TRUE(utils::cerial_testing(add1));
}