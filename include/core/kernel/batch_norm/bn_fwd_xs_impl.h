//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BN_FWD_XS_IMPL_H
#define XSDNN_BN_FWD_XS_IMPL_H

#include "../../framework/params.h"
#include "../../../utils/tensor.h"

namespace xsdnn {
    namespace kernel {

        void batch_normalization_fwd_xs_impl(const tensor_t& in,
                                             const mat_t& gamma,
                                             const mat_t& beta,
                                             tensor_t& out,
                                             params::bnorm& p,
                                             bool parallelize,
                                             size_t nthreads);

    } // kernel
} // xsdnn

#endif //XSDNN_BN_FWD_XS_IMPL_H
