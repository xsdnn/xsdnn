//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_CORE_H
#define XSDNN_CORE_H

#include <iostream>
#include "../Utils/Except.h"

// TODO: продумать логику ядра для рекуррентных сетей

namespace core {
    enum class conv2d { direct, mec, im2col, fft };
    inline std::ostream& operator << (std::ostream& out, conv2d& obj) {
        switch (obj) {
            case conv2d::direct:
                out << "Directly Convolution Algorithm";
                break;

            case conv2d::mec:
                out << "Memory Efficient Convolution Algorithm (MEC)";
                break;

            case conv2d::im2col:
                out << "Im2Col-based Convolution Algorithm";
                break;

            case conv2d::fft:
                out << "Fast Fourier Transform based Convolution Algorithm";
                break;

            default:
                throw internal::except::xs_error("Undefined conv enum ostream");
        }
        return out;
    }

    inline conv2d default_core() {
        return conv2d::mec;
    }
} // end namespace core

#endif //XSDNN_CORE_H
