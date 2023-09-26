//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../include/core/framework/allocator.h"
#include "../include/core/framework/tensor.h"
using namespace xsdnn;
#include <gtest/gtest.h>

TEST(Tensor, DefaultCPUAlloc) {
    CPUAllocator CPUAlloc;
    void* ptr = CPUAlloc.Alloc(10);
    ASSERT_TRUE(ptr != nullptr);
}

TEST(Tensor, Shape) {
//    tensor_shape shape1({1, 3, 3, 3});
//    ASSERT_EQ(shape1[0], 1);
//    ASSERT_EQ(shape1[1], 3);
//    ASSERT_EQ(shape1[2], 3);
//    ASSERT_EQ(shape1[3], 3);
//    ASSERT_EQ(shape1.num_dimensions(), 4);
//    ASSERT_EQ(shape1.size(), 27);
//
//    tensor_shape shape2({1, 3, 9});
//    ASSERT_EQ(shape2[0], 1);
//    ASSERT_EQ(shape2[1], 3);
//    ASSERT_EQ(shape2[2], 9);
//    ASSERT_EQ(shape2.num_dimensions(), 3);
//    ASSERT_EQ(shape1.size(), 27);
//
//
//    tensor_shape shape3 = shape1.slice(1);
//    ASSERT_EQ(shape3[0], 3);
//    ASSERT_EQ(shape3[1], 3);
//    ASSERT_EQ(shape3[2], 3);
//    ASSERT_EQ(shape3.num_dimensions(), 3);
//    ASSERT_EQ(shape1.size(), 27);
}

TEST(Tensor, Ctor1) {
//    CPUAllocator CPUAlloc;
//    tensor_t Tensor(XsDtype::F32, tensor_shape({1, 3, 28, 28}), &CPUAlloc);
//    gsl::span<uint8_t> TensorData = Tensor.GetMutableDataAsSpan<uint8_t>();
}
