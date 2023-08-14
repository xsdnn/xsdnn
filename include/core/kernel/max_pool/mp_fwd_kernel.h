//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MP_FWD_KERNEL_H
#define XSDNN_MP_FWD_KERNEL_H

#include "../../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

class MaxPoolingFwdKernel : public OpKernel {
public:
    void compute(OpContext& ctx, params::max_pool& p);
};

    } // core
} // xsdnn

#endif //XSDNN_MP_FWD_KERNEL_H
