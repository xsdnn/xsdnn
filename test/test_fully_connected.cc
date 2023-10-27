//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(fc, forward_fp32) {
    fully_connected fc(6, 3);
    fc.weight_init(weight_init::constant(0.0f));
    fc.bias_init(weight_init::constant(1.0f));

    mat_t in(6 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in, {0, 1, 2, 3, 4, 5});

    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];

    mat_t e(3 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(e, {1, 1, 1});

    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&o);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&e);

    for (size_t i = 0; i < e.size() / sizeof(float); i++) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(fc, forward_nobias_fp32) {
    fully_connected fc(6, 3, false);
    fc.weight_init(weight_init::constant(0.5f));

    mat_t in(6 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ in }});
    fc.forward();
    mat_t o = fc.output()[0][0];

    mat_t e(3 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(e, {7.5f, 7.5f, 7.5f});

    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&o);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&e);

    for (size_t i = 0; i < e.size() / sizeof(float); i++) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(fc, cerial) {
    fully_connected fc(50, 100);
    ASSERT_TRUE(utils::cerial_testing(fc));
}