//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

// TODO: impl this

TEST(max_pool, forward) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{ in_data }});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {5, 6, 7, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
        #error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_stride_x) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 1, 2);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {5, 6, 6, 7, 4, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_stride_y) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 2, 1);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {5, 6, 5, 2, 7, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_stride_xy) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 1, 1);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {5, 6, 6, 5, 5, 2, 7, 4, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_kernel_x) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 4, 2, 2, 2);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {6, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_kernel_x2) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 3, 2, 2, 2);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {6, 7};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_kernel_x3_padding_same) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 3, 2, 2, 2, padding_mode::same);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {6, 6, 7, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_kernel_y) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 3, 2, 2);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {5, 6};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_kernel_y2) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 4, 2, 2);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {7, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}

TEST(max_pool, forward_kernel_y3_padding_same) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 3, 2, 2, padding_mode::same);
    mat_t in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp = {5, 6, 7, 8};
    ASSERT_TRUE(out.size() == exp.size());
    for (size_t i = 0; i < out.size(); ++i) {
#ifdef MM_USE_DOUBLE
#error NotImpl
#else
        ASSERT_FLOAT_EQ(out[i], exp[i]);
#endif
    }
}




