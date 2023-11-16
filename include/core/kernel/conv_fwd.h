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
            void Compute(OpContext& ctx, params::conv& p) override;
#ifdef XS_USE_XNNPACK
            void CreateAndReshapeXNNKernel(xsDtype dtype, std::vector<mat_t*> WB, params::conv& p) override;

        private:
            std::vector<char> workspace;

            size_t max_workspace_size{0};
            size_t workspace_size{0};
            size_t workspace_alignment{0};
#endif


        };

    } // core
} // xsdnn

#endif //XSDNN_CONV_FWD_H
