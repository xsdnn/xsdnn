//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(add, forward) {

}

TEST(add, cerial) {
    shape3d shape_ = shape3d(3, 28, 28);
    xsdnn::and_layer and_(shape_);
    ASSERT_TRUE(utils::cerial_testing(and_));
}
