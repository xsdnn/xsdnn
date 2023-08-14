//
// Created by rozhin on 07.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>

TEST(batch_norm, simple_forward) {
    xsdnn::batch_norm bn;

    xsdnn::mat_t in_data = {0, 1, 2, 3, 4, 5};
    bn.set_in_shape(xsdnn::shape3d(1, 1, 6)); // [C, H, W]
    bn.set_in_data({{ in_data }});
    bn.setup(false);
    bn.set_parallelize(false);
    bn.forward();

    xsdnn::mat_t out = bn.output()[0][0];
    xsdnn::mat_t ex = {-1.3363043 , -0.8017826 , -0.26726085,  0.26726085,  0.8017826 ,
                       1.3363043};

    for (size_t i = 0; i < 6; ++i) {
#ifdef MM_USE_DOUBLE
#error NotImplementedYet
#else
        ASSERT_FLOAT_EQ(out[i], ex[i]);
#endif
    }
}