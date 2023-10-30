//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include "test_utils.h"
using namespace xsdnn;

TEST(output, forward_fp32) {
    shape3d shape_ = shape3d(3, 224, 224);
    Output out(shape_);
    mat_t in_data(shape_.size() * dtype2sizeof(kXsFloat32));
    utils::random_init_fp32(in_data);

    out.set_in_data({{ in_data }});
    out.set_parallelize(false);
    out.setup(false);

    out.forward();
    mat_t out_ = out.output()[0][0];
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out_);
    gsl::span<const float> InSpan = GetDataAsSpan<const float>(&in_data);

    for (size_t h = 0; h < shape_.H; ++h) {
        for (size_t w = 0; w < shape_.W; ++w) {
            for (size_t c = 0; c < shape_.C; ++c) {
                utils::xsAssert_eq(OutSpan[shape_(c, h, w)], InSpan[shape_(c, h, w)], kXsFloat32);
            }
        }
    }
}

#ifdef XS_USE_SERIALIZATION
TEST(output, cerial) {
    shape3d shape_ = shape3d(3, 224, 224);
    Output out(shape_);
    ASSERT_TRUE(utils::cerial_testing(out));
}
#endif
