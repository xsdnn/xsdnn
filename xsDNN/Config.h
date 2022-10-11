//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
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

namespace xsdnn {
    namespace internal {
        int64_t num_core() {
#if defined(_WIN32)
            SYSTEM_INFO sysinfo;
            GetSystemInfo(&sysinfo);
            return sysinfo.dwNumberOfProcessors;
#elif defined(__linux__) || defined(unix)
            return sysconf(_SC_NPROCESSORS_ONLN);
#endif
        }
    } // namespace internal

    namespace xsThread {
        Eigen::ThreadPool pool(internal::num_core());
        Eigen::ThreadPoolDevice CPUDevice(&pool, pool.NumThreads());
    } // namespace xsThread
} // namespace xsdnn

#endif //XSDNN_CONFIG_H
