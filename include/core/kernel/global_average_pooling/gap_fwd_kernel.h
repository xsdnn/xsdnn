//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_GAP_FWD_KERNEL_H
#define XSDNN_GAP_FWD_KERNEL_H

#include "../../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

        class GlobalAvgPoolingFwdKernel : public OpKernel {
        public:
            void compute(OpContext& ctx, params::global_avg_pool& p);
        };

    } // core
} // xsdnn

#endif //XSDNN_GAP_FWD_KERNEL_H
