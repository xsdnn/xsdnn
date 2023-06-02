//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_PARAMS_H
#define XSDNN_PARAMS_H

#include <cstddef>

namespace xsdnn {
    namespace params {

struct fully {
    size_t in_size_;
    size_t out_size_;
    bool   has_bias_;
};

    } // params
} // xsdnn

#endif //XSDNN_PARAMS_H
