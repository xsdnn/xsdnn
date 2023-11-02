//
// Created by rozhin on 02.11.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../include/utils/transpose.h"
#include "test_utils.h"

TEST(SimpleTransposeSingleAxisFP32, Matrix2D) {
    shape3d XShape(1, 2, 3);
    mat_t X(XShape.size() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(X, {1, 2, 3, 4, 5, 6});

    mat_t TransposeX(XShape.size() * dtype2sizeof(kXsFloat32));
    xsdnn::xs_single_axis_transpose(&X, {1, 2, 3}, &TransposeX, kXsFloat32, 1, 2);

    gsl::span<const float> XSpan = GetDataAsSpan<const float>(&X);
    gsl::span<const float> TransposeXSpan = GetDataAsSpan<const float>(&TransposeX);

    utils::xsAssert_eq(TransposeXSpan[0], 1.0f, kXsFloat32);
    utils::xsAssert_eq(TransposeXSpan[1], 4.0f, kXsFloat32);
    utils::xsAssert_eq(TransposeXSpan[2], 2.0f, kXsFloat32);
    utils::xsAssert_eq(TransposeXSpan[3], 5.0f, kXsFloat32);
    utils::xsAssert_eq(TransposeXSpan[4], 3.0f, kXsFloat32);
    utils::xsAssert_eq(TransposeXSpan[5], 6.0f, kXsFloat32);
}