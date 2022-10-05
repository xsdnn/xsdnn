//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_SOME_STUPID_TEST_H
#define XSDNN_SOME_STUPID_TEST_H

TEST(tensor, map) {
    typedef Eigen::TensorMap<Eigen::Tensor<Scalar, 3, Eigen::ColMajor, Eigen::DenseIndex>, Eigen::Aligned>
            Tensor;

    Eigen::Tensor<Scalar, 3, Eigen::ColMajor, Eigen::DenseIndex> storage;
    Tensor a(storage.data(), 2, 4, 2);
}

#endif //XSDNN_SOME_STUPID_TEST_H
