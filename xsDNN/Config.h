//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_CONFIG_H
#define XSDNN_CONFIG_H

namespace xsdnn {
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
} // namespace xsdnn

// TODO: подготовить план по рефакторингу кода под Eigen::Tensor

namespace xsdnn {
    template<typename T, int NDIMS = 2>
    struct xsTypes {
        typedef Eigen::Tensor<T, NDIMS, Eigen::ColMajor, Eigen::DenseIndex>
                Tensor;
        typedef Eigen::Tensor<const T, NDIMS, Eigen::ColMajor, Eigen::DenseIndex>
                ConstTensor;
    };
} // namespace xsdnn

#endif //XSDNN_CONFIG_H
