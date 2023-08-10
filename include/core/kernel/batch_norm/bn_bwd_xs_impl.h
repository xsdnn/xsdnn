//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BN_BWD_XS_IMPL_H
#define XSDNN_BN_BWD_XS_IMPL_H

#include "../../framework/params.h"
#include "../../../utils/tensor.h"

namespace xsdnn {
    namespace kernel {

        void batch_normalization_bwd_xs_impl(const tensor_t& in,
                                             const tensor_t& out,
                                             tensor_t& dx,
                                             tensor_t& dg,
                                             tensor_t& db,
                                             const tensor_t& dlz,
                                             params::bnorm& p,
                                             bool parallelize,
                                             size_t nthreads);

    } // kernel
} // xsdnn

#endif //XSDNN_BN_BWD_XS_IMPL_H
