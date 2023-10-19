//
// Created by rozhin on 19.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_GLOBAL_AVG_POOL_FWD_H
#define XSDNN_GLOBAL_AVG_POOL_FWD_H

#include "../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

        class GlobalAvgPoolingFwdKernel : public OpKernel {
        public:
            void Compute(OpContext& ctx, params::global_avg_pool& p);
        };

    } // core
} // xsdnn

#endif //XSDNN_GLOBAL_AVG_POOL_FWD_H
