//
// Created by Andrei R. on 13.10.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_CONFIG_H
#define XSDNN_CONFIG_H


#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__) || defined(unix)
#include <unistd.h>
#else
printf("Unsupported OS");
exit(1);
#endif

#if defined(DNN_USE_DOUBLE)
typedef double Scalar;
#else
typedef float Scalar;
#endif

namespace xsdnn {
    struct xsTypes {
        typedef Eigen::TensorMap<Eigen::Tensor<Scalar, 4, Eigen::ColMajor, Eigen::DenseIndex>, Eigen::Aligned>
                AlignedTensor_4D;
        typedef Eigen::TensorMap<Eigen::Tensor<Scalar, 3, Eigen::ColMajor, Eigen::DenseIndex>, Eigen::Aligned>
                AlignedTensor_3D;
        typedef Eigen::TensorMap<Eigen::Tensor<Scalar, 2, Eigen::ColMajor, Eigen::DenseIndex>, Eigen::Aligned>
                AlignedMatrix;
        typedef Eigen::TensorMap<Eigen::Tensor<Scalar, 1, Eigen::ColMajor, Eigen::DenseIndex>, Eigen::Aligned>
                AlignedVector;
    };

    // use for up-level API

    // (N, C, H, W)
    typedef Eigen::Tensor<Scalar, 4, Eigen::ColMajor, Eigen::DenseIndex>
            Tensor_4D;

    // (C, H, W)
    typedef Eigen::Tensor<Scalar, 3, Eigen::ColMajor, Eigen::DenseIndex>
            Tensor_3D;

    // (H, W)
    typedef Eigen::Tensor<Scalar, 2, Eigen::ColMajor, Eigen::DenseIndex>
            Matrix;

    // (W)
    typedef Eigen::Tensor<Scalar, 1, Eigen::ColMajor, Eigen::DenseIndex>
            Vector;

    namespace xsThread {
        int64_t num_core() {
#if defined(_WIN32)
            SYSTEM_INFO sysinfo;
            GetSystemInfo(&sysinfo);
            return sysinfo.dwNumberOfProcessors;
#elif defined(__linux__) || defined(unix)
            return sysconf(_SC_NPROCESSORS_ONLN);
#endif
        }
        Eigen::ThreadPool pool(num_core());
        Eigen::ThreadPoolDevice cpu(&pool, pool.NumThreads());
    } // namespace xsThread
} // namespace xsdnn

#endif //XSDNN_CONFIG_H
