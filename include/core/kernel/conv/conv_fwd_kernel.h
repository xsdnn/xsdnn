//
// Created by rozhin on 23.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_CONV_FWD_KERNEL_H
#define XSDNN_CONV_FWD_KERNEL_H

#include "../../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

class ConvFwdKernel : public OpKernel {
public:
    virtual void compute(OpContext &ctx, params::conv &p);
};

    } // core
} // xsdnn

#endif //XSDNN_CONV_FWD_KERNEL_H
