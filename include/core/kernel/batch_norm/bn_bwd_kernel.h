//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BN_BWD_KERNEL_H
#define XSDNN_BN_BWD_KERNEL_H

#include "../../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

        class BatchNormalizationBwdKernel : public OpKernel {
        public:
            virtual void compute(OpContext& ctx, params::bnorm& p);
        };

    } // core
} // xsdnn

#endif //XSDNN_BN_BWD_KERNEL_H
