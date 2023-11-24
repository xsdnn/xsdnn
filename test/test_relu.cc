//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(relu, forward_fp32) {
    relu rl(5);
    mat_t in;
    AllocateMat_t(&in, 5, kXsFloat32);
    utils::initializer_list_init_fp32(in, {-1, -0.5, 0.0, 0.5, 1});

    rl.setup(false);
    rl.set_parallelize(false);
    rl.prepare();

    rl.set_in_data({{ in }});
    rl.forward();

    mat_t out = rl.output()[0][0];
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);

    mat_t e;
    AllocateMat_t(&e, 5, kXsFloat32);
    utils::initializer_list_init_fp32(e, {0.0f, 0.0f, 0.0f, 0.5, 1.0f});
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&e);

    for (size_t i = 0; i < 5; i++) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}
#ifdef XS_USE_SERIALIZATION
TEST(relu, cerial) {
    relu rl(784);
    ASSERT_TRUE(utils::cerial_testing(rl));
}
#endif