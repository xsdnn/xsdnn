//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(global_average_pooling, forward) {
    shape3d in_shape(1, 4, 4);
    global_average_pooling pool(in_shape);

    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{ in_data }});
    pool.forward();

    const auto out = pool.output()[0][0];
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
    ASSERT_FLOAT_EQ(out[0], 3.1875f);
#endif
}

TEST(global_average_pooling, forward_two_channels) {
    shape3d in_shape(2, 4, 4);
    global_average_pooling pool(in_shape);

    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8,
                     1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{ in_data }});
    pool.forward();

    const auto out = pool.output()[0][0];
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
    ASSERT_FLOAT_EQ(out[0], 3.1875f);
    ASSERT_FLOAT_EQ(out[1], 3.1875f);
#endif
}

TEST(global_average_pooling, cerial) {
    shape3d in_shape(3, 224, 224);
    global_average_pooling pool(in_shape);
    ASSERT_TRUE(utils::cerial_testing(pool));
}