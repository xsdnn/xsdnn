//
// Created by rozhin on 11.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;

TEST(Output, Forward_F32) {
    shape3d shape_(3, 224, 224);
    Output in(shape_);
    std::vector<float> in_data(shape_.size());
    utils::random_init(in_data.data(), shape_.size());

    tensor_t Tensor(XsDtype::F32, shape_, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    in.set_in_data({{ Tensor }});
    in.set_parallelize(false);
    in.setup(false);
    in.forward();

    tensor_t OutTensor = in.output()[0][0];
    tensor_t ExpectedTensor(XsDtype::F32, shape_, nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), in_data);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(Output, Cerial) {
    shape3d shape_ = shape3d(3, 224, 224);
    Output in(shape_);
    ASSERT_TRUE(utils::cerial_testing(in));
}

