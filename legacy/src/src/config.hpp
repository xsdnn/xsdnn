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

#include "util/xs_error.hpp"

namespace xsdnn {
#if defined(XS_USE_DOUBLE)
    typedef double Scalar;
#else
    typedef float Scalar;
#endif

#ifdef XS_USE_LOG
namespace logger {

enum class type {
    info,
    warn,
    error,
    critical
};

std::string logger_name = "default_logger";
std::string logger_path = "logs/log.txt";

auto file_logger = spdlog::basic_logger_mt(logger_name, logger_path);

void log(type type, const std::string& msg) {
    switch(type) {
        case type::info:
            spdlog::info(msg);
            file_logger->info(msg);
            break;

        case type::warn:
            spdlog::warn(msg);
            file_logger->warn(msg);
            break;

        case type::error:
            spdlog::error(msg);
            file_logger->error(msg);
            break;

        case type::critical:
            spdlog::critical(msg);
            file_logger->critical(msg);
            break;

        default:
            throw xs_error("Unsupported logger type");
    }
}

} // logger
#endif

    using json = nlohmann::json;
    using Index = Eigen::DenseIndex;

    // (N, C, H, W)
    typedef Eigen::Tensor<Scalar, 4, Eigen::AutoAlign, Index>
            Tensor_4D;

    // (C, H, W)
    typedef Eigen::Tensor<Scalar, 3, Eigen::AutoAlign, Index>
            Tensor_3D;

    // (H, W)
    typedef Eigen::Tensor<Scalar, 2, Eigen::AutoAlign, Index>
            Matrix;

    // (W)
    typedef Eigen::Tensor<Scalar, 1, Eigen::AutoAlign, Index>
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
