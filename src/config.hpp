//
// Created by Andrei R. on 13.10.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_CONFIG_HPP
#define XSDNN_CONFIG_HPP


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
// (N, C, H, W)
    typedef Eigen::Tensor<Scalar, 4, Eigen::AutoAlign, Eigen::DenseIndex>
            Tensor_4D;

// (C, W, H)
    typedef Eigen::Tensor<Scalar, 3, Eigen::AutoAlign, Eigen::DenseIndex>
            Tensor_3D;

// (W, H)
    typedef Eigen::Tensor<Scalar, 2, Eigen::AutoAlign, Eigen::DenseIndex>
            Matrix;

// (H)
    typedef Eigen::Tensor<Scalar, 1, Eigen::AutoAlign, Eigen::DenseIndex>
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
    } // xsThread
} // xsdnn

#endif //XSDNN_CONFIG_HPP
