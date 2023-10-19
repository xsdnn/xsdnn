//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_OP_KERNEL_H
#define XSDNN_OP_KERNEL_H

#include "op_context.h"
#include "params.h"

namespace xsdnn {
    namespace core {

class OpKernel {
public:
    OpKernel() = default;
    OpKernel(const OpKernel&) = delete;
    OpKernel& operator=(const OpKernel&) = delete;
    virtual ~OpKernel() {}

public:
    virtual void Compute(core::OpContext& ctx, params::fully& p) {}
    virtual void Compute(core::OpContext& ctx, params::max_pool& p) {}
    virtual void Compute(core::OpContext& ctx, params::conv& p) {}
};

    } // core
} // xsdnn

#endif //XSDNN_OP_KERNEL_H
