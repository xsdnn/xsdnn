//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_OP_KERNEL_H
#define XSDNN_OP_KERNEL_H

#include "op_context.h"
#include "params.h"
#ifdef XS_USE_PROFILING_TOOLS
#include <easy/profiler.h>
#endif

namespace xsdnn {
    namespace core {

class OpKernel {
public:
    OpKernel() = default;
    OpKernel(const OpKernel&) = delete;
    OpKernel& operator=(const OpKernel&) = delete;
    virtual ~OpKernel() {}

#ifdef XS_USE_XNNPACK
public:
    virtual void CreateAndReshapeXNNKernel(xsDtype dtype, std::vector<mat_t*> WB, params::fully& p) {}
    virtual void CreateAndReshapeXNNKernel(xsDtype dtype, std::vector<mat_t*> WB, params::max_pool& p) {}
    virtual void CreateAndReshapeXNNKernel(xsDtype dtype, std::vector<mat_t*> WB, params::conv& p) {}
    virtual void CreateAndReshapeXNNKernel(xsDtype dtype, std::vector<mat_t*> WB, params::activation& p) {}
#endif

public:
    /*
     * Runtime methods
     */
    virtual void Compute(core::OpContext& ctx, params::fully& p) {}
    virtual void Compute(core::OpContext& ctx, params::max_pool& p) {}
    virtual void Compute(core::OpContext& ctx, params::conv& p) {}
    virtual void Compute(core::OpContext& ctx, params::activation& p) {}

#ifdef XS_USE_XNNPACK
protected:
    xnn_operator_t op_;
#endif
};

    } // core
} // xsdnn

#endif //XSDNN_OP_KERNEL_H
