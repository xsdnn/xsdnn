//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BN_FWD_XS_IMPL_H
#define XSDNN_BN_FWD_XS_IMPL_H

#include "../../framework/params.h"
#include "../../../core/framework/tensor.h"

namespace xsdnn {
    namespace kernel {

        void batch_normalization_fwd_xs_impl(const BTensor& in,
                                             const tensor_t& gamma,
                                             const tensor_t& beta,
                                             BTensor& out,
                                             params::bnorm& p,
                                             bool parallelize,
                                             size_t nthreads);

    } // kernel
} // xsdnn

#endif //XSDNN_BN_FWD_XS_IMPL_H
