//
// Created by rozhin on 24.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_ACTIVATION_FWD_H
#define XSDNN_ACTIVATION_FWD_H

#include "../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

        class ActivationFwdKernel : public OpKernel {
        public:
            void Compute(OpContext& ctx, params::activation& p);
        };

    } // core
} // xsdnn
#endif //XSDNN_ACTIVATION_FWD_H
