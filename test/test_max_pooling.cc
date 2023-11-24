//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;


TEST(max_pool, forward_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
                     1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{ in_data }});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(4 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {5, 6, 7, 8});
    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_stride_x_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 1, 2);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(6 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {5, 6, 6, 7, 4, 8});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_stride_y_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 2, 1);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(6 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {5, 6, 5, 2, 7, 8});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_stride_xy_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 1, 1);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(9 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {5, 6, 6, 5, 5, 2, 7, 4, 8});
    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_kernel_x_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 4, 2, 2, 2);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(2 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {6, 8});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_kernel_x2_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 3, 2, 2, 2);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(2 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {6, 7});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_kernel_x3_padding_same_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 3, 2, 2, 2, padding_mode::same);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(4 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {6, 6, 7, 8});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_kernel_y_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 3, 2, 2);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    mat_t out = pool.output()[0][0];
    mat_t exp(2 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {5, 6});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_kernel_y2_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 4, 2, 2);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp(2 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {7, 8});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}

TEST(max_pool, forward_kernel_y3_padding_same_fp32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 3, 2, 2, padding_mode::same);
    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(in_data, {
            1, 2, 6, 3,
            3, 5, 2, 1,
            1, 2, 2, 1,
            7, 3, 4, 8
    });

    pool.setup(false);
    pool.set_parallelize(false);
    pool.prepare();

    pool.set_in_data({{in_data}});
    pool.forward();

    const auto out = pool.output()[0][0];
    mat_t exp(4 * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(exp, {5, 6, 7, 8});

    ASSERT_TRUE(out.size() == exp.size());
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&exp);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}
#ifdef XS_USE_SERIALIZATION
TEST(max_pool, cerial) {
    shape3d in_shape(3, 224, 224);
    max_pooling pool(in_shape, 14, 28, 3, 8, padding_mode::same);
    ASSERT_TRUE(utils::cerial_testing(pool));
}
#endif




