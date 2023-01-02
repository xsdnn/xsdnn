//
// Created by Andrei R. on 02.01.23.
// Copyright (c) 2023 xsDNN. All rights reserved.
//

#ifndef XSDNN_BACKEND_HPP
#define XSDNN_BACKEND_HPP

namespace xsdnn {
namespace core {

enum class backend_t {
    default_cpu,
    cuda
};

std::ostream& operator<<(std::ostream& os, backend_t obj) {
    switch (obj) {
        case backend_t::default_cpu:
            os << "Default CPU - Eigen";
            break;
        case backend_t::cuda:
            os << "CUDA";
            break;
        default:
            throw xs_error("Unsupported backend");
    }
    return os;
}

} // core
} // xsdnn

#endif //XSDNN_BACKEND_HPP
