//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include "test_utils.h"
using namespace xsdnn;


TEST(MaxPool, Forward1_F32) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2);
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
    std::vector<float> e = {5, 6, 7, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 4), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_stride_x) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 1, 2);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {5, 6, 6, 7, 4, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 6), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_stride_y) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 2, 1);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {5, 6, 5, 2, 7, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 6), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_stride_xy) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 2, 1, 1);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {5, 6, 6, 5, 5, 2, 7, 4, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 9), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_kernel_x) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 4, 2, 2, 2);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {6, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 2), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_kernel_x2) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 3, 2, 2, 2);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {6, 7};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 2), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_kernel_x3_padding_same) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 3, 2, 2, 2, padding_mode::same);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {6, 6, 7, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 4), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_kernel_y) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 3, 2, 2);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {5, 6};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 2), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_kernel_y2) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 4, 2, 2);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {7, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 2), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, forward_kernel_y3_padding_same) {
    shape3d in_shape(1, 4, 4);
    max_pooling pool(in_shape, 2, 3, 2, 2, padding_mode::same);
    std::vector<float> in_data = {1, 2, 6, 3,
                     3, 5, 2, 1,
                     1, 2, 2, 1,
                     7, 3, 4, 8};
    tensor_t Tensor(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(Tensor.GetMutableData<float>(), in_data);

    pool.setup(false);
    pool.set_parallelize(false);
    pool.set_in_data({{Tensor}});
    pool.forward();

    tensor_t OutTensor = pool.output()[0][0];
    std::vector<float> e = {5, 6, 7, 8};
    tensor_t ExpectedTensor(XsDtype::F32, shape3d(1, 1, 4), nullptr);
    utils::vector_init(ExpectedTensor.GetMutableData<float>(), e);
    utils::ContainerEqual(OutTensor, ExpectedTensor);
}

TEST(max_pool, cerial) {
    shape3d in_shape(3, 224, 224);
    max_pooling pool(in_shape, 14, 28, 3, 8, padding_mode::same);
    ASSERT_TRUE(utils::cerial_testing(pool));
}




