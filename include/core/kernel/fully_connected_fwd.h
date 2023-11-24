//
// Created by rozhin on 19.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FULLY_CONNECTED_FWD_H
#define XSDNN_FULLY_CONNECTED_FWD_H

#include "../framework/op_kernel.h"

namespace xsdnn {
    namespace core {

        class FullyConnectedFwdKernel : public OpKernel {
        public:
            virtual void Compute(OpContext& ctx, params::fully& p);
#ifdef XS_USE_XNNPACK
            void CreateAndReshapeXNNKernel(xsDtype dtype, std::vector<mat_t*> WB, params::fully& p) override;
#endif
        };

    } // core
} // xsdnn

#endif //XSDNN_FULLY_CONNECTED_FWD_H
