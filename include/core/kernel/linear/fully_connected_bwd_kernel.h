//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FULLY_CONNECTED_BWD_KERNEL_H
#define XSDNN_FULLY_CONNECTED_BWD_KERNEL_H

#include <core/framework/op_kernel.h>

namespace xsdnn {
    namespace core {

class FullyConnectedBwdKernel : public OpKernel {
public:
    virtual void compute(OpContext& ctx, params::fully& p);
};

    } // core
} // xsdnn

#endif //XSDNN_FULLY_CONNECTED_BWD_KERNEL_H
