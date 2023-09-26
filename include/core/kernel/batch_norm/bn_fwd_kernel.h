//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BN_FWD_KERNEL_H
#define XSDNN_BN_FWD_KERNEL_H

#include "../../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

        class BatchNormalizationFwdKernel : public OpKernel {
        public:
            virtual void compute(OpContext& ctx, params::bnorm& p);

        private:
            void init_statistics(params::bnorm& p, XsDtype TensorDType);
        };

    } // core
} // xsdnn

#endif //XSDNN_BN_FWD_KERNEL_H
