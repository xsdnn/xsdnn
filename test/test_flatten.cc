//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(flatten, forward_fp32) {
    shape3d in_shape(3, 128, 224);
    flatten fl(in_shape);

    mat_t in_data(in_shape.size() * dtype2sizeof(kXsFloat32));
    utils::random_init_fp32(in_data);

    fl.set_in_data({{ in_data }});
    fl.set_parallelize(false);
    fl.setup(false);

    fl.forward();
    ASSERT_TRUE(fl.out_shape()[0] == shape3d(1, 1, in_shape.size()));
}

TEST(flatten, cerial) {
    shape3d in_shape(128, 224, 3);
    flatten fl(in_shape);
    ASSERT_TRUE(utils::cerial_testing(fl));
}