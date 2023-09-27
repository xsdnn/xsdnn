//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(ReluActivation, Forward_F32) {
    relu rl(5);

    std::vector<float> in = {-1, -0.5, 0.0, 0.5, 1};
    tensor_t Tensor(XsDtype::F32, shape3d(1, 1, 5), nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in);

    rl.setup(false);
    rl.set_parallelize(false);
    rl.set_in_data({{ Tensor }});
    rl.forward();

    tensor_t OutTensor = rl.output()[0][0];
    std::vector<float> e = {0.0f, 0.0f, 0.0f, 0.5, 1.0f};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 5), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(ReluActivation, Cerial) {
    relu rl(784);
    ASSERT_TRUE(utils::cerial_testing(rl));
}