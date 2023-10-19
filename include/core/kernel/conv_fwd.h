//
// Created by rozhin on 19.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_CONV_FWD_H
#define XSDNN_CONV_FWD_H

#include "../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

        class ConvFwdKernel : public OpKernel {
        public:
            void compute(OpContext& ctx, params::conv& p);
        };

    } // core
} // xsdnn

#endif //XSDNN_CONV_FWD_H
