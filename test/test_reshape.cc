//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

#ifdef XS_USE_SERIALIZATION
TEST(reshape, cerial) {
    shape3d shape_ = shape3d(3, 224, 224);
    reshape res(shape_);
    ASSERT_TRUE(utils::cerial_testing(res));
}
#endif