//
// Created by rozhin on 21.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(conv, simple_2D) {
    shape3d in_shape(3, 28, 28);
    conv c(in_shape, 16, 1, true, {3, 3});
}

TEST(conv, simple_1D) {
    shape3d in_shape(1, 1, 728);
    conv c(in_shape, 16);
}