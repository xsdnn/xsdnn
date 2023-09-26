//
// Created by rozhin on 07.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "test_utils.h"
#include "xsdnn.h"
#include <gtest/gtest.h>
using namespace xsdnn;

// TODO: как устанавливать веса?

TEST(batch_norm, simple_forward) {
    batch_norm bn;
    shape3d InShape(1, 1, 6);
    std::vector<float> InDataVector = {0, 1, 2, 3, 4, 5};
    tensor_t InTensor(XsDtype::F32, InShape, nullptr);
    utils::vector_init(InTensor.GetMutableData<float>(), InDataVector);

    bn.set_in_shape(xsdnn::shape3d(1, 1, 6));
    bn.set_in_data({{ InTensor }});
    bn.setup(false);
    bn.set_parallelize(false);
    bn.forward();

    tensor_t out = bn.output()[0][0];
    std::vector<float> ex = {-1.3363043 , -0.8017826 , -0.26726085,  0.26726085,  0.8017826 ,
                       1.3363043};
    tensor_t ExpectedTensor(XsDtype::F32, InShape, nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), ex);
    utils::ContainerEqual(out, ExpectedTensor);
}