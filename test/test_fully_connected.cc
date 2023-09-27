//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(FullyConnected, Forward1_F32) {
    fully_connected fc(6, 3);
    fc.weight_init(weight_init::constant(0.0f));
    fc.bias_init(weight_init::constant(1.0f));

    std::vector<float> in = {0, 1, 2, 3, 4, 5};
    tensor_t Tensor(XsDtype::F32, shape3d(1, 1, 6), nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in);

    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ Tensor }});
    fc.forward();

    tensor_t OutTensor = fc.output()[0][0];
    std::vector<float> e = {1, 1, 1};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(FullyConnected, Forward2_F32) {
    fully_connected fc(6, 3);
    fc.weight_init(weight_init::constant(0.5f));
    fc.bias_init(weight_init::constant(0.3f));

    std::vector<float> in = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    tensor_t Tensor(XsDtype::F32, shape3d(1, 1, 6), nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in);

    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ Tensor }});
    fc.forward();

    tensor_t OutTensor = fc.output()[0][0];
    std::vector<float> e = {7.8f, 7.8f, 7.8f};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(FullyConnected, ForwardNoBias_F32) {
    fully_connected fc(6, 3, false);
    fc.weight_init(weight_init::constant(0.5f));

    std::vector<float> in = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    tensor_t Tensor(XsDtype::F32, shape3d(1, 1, 6), nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in);

    fc.set_parallelize(false);
    fc.setup(false);
    fc.set_in_data({{ Tensor }});
    fc.forward();

    tensor_t OutTensor = fc.output()[0][0];
    std::vector<float> e = {7.5f, 7.5f, 7.5f};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(fc, cerial) {
    fully_connected fc(50, 100);
    ASSERT_TRUE(utils::cerial_testing(fc));
}