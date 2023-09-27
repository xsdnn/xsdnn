//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(global_average_pooling, forward) {
    shape3d in_shape(1, 4, 4);
    global_average_pooling pool(in_shape);

    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{ Tensor }});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {3.1875f};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 1), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(global_average_pooling, forward_two_channels) {
    shape3d in_shape(2, 4, 4);
    global_average_pooling pool(in_shape);

    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8,
                     1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{ Tensor }});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {3.1875f, 3.1875f};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 2), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(global_average_pooling, cerial) {
    shape3d in_shape(3, 224, 224);
    global_average_pooling pool(in_shape);
    ASSERT_TRUE(utils::cerial_testing(pool));
}