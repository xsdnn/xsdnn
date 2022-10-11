//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_CONFIG_H
#define XSDNN_CONFIG_H


namespace xsdnn {
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
} // namespace xsdnn

// TODO: сделать ванильный рефакторинг всего кода

namespace xsdnn {
    struct xsTypes {
        typedef Eigen::Tensor<Scalar, 4, Eigen::ColMajor, Eigen::DenseIndex>
                Tensor_4D;
        typedef Eigen::Tensor<Scalar, 3, Eigen::ColMajor, Eigen::DenseIndex>
                Tensor_3D;
        typedef Eigen::Tensor<Scalar, 2, Eigen::ColMajor, Eigen::DenseIndex>
                Matrix;
        typedef Eigen::Tensor<Scalar, 1, Eigen::ColMajor, Eigen::DenseIndex>
                Vec;
    };
} // namespace xsdnn

// TODO: доделать поточную структуру с пулом и девайсом
namespace xsdnn {
    struct xsThread {
        sysconf(_SC_NPROCESSORS_ONLN);
        Eigen::ThreadPool pool()
    };
} // namespace xsdnn

#endif //XSDNN_CONFIG_H
