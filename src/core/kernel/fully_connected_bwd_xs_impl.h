//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FULLY_CONNECTED_BWD_XS_IMPL_H
#define XSDNN_FULLY_CONNECTED_BWD_XS_IMPL_H

#include "../../utils/tensor.h"
#include "../../utils/macro.h"
#include "../framework/params.h"

namespace xsdnn {
    namespace kernel {

void fully_connected_bwd_xs_impl(const tensor_t& x,
                                 const mat_t& W,
                                 tensor_t& dx,
                                 tensor_t& dW,
                                 tensor_t& db,
                                 tensor_t& dLz,
                                 const params::fully& p,
                                 bool parallelize
                                 );

    } // kernel
} // xsdnn

#endif //XSDNN_FULLY_CONNECTED_BWD_XS_IMPL_H
