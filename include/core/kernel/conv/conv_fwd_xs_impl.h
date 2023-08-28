//
// Created by rozhin on 23.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_CONV_FWD_XS_IMPL_H
#define XSDNN_CONV_FWD_XS_IMPL_H

#include "../../framework/params.h"
#include "../../../utils/tensor.h"

namespace xsdnn {
    namespace kernel {

        void conv_fwd_xs_impl(const tensor_t& X,
                            const mat_t& W,
                            const mat_t* B,
                            tensor_t& Y,
                            params::conv& p,
                            bool parallelize,
                            size_t nthreads);

    }
}

#endif //XSDNN_CONV_FWD_XS_IMPL_H
