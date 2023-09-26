//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FULLY_CONNECTED_FWD_XS_IMPL_H
#define XSDNN_FULLY_CONNECTED_FWD_XS_IMPL_H

#include "../../framework/params.h"
#include "../../framework/tensor.h"

namespace xsdnn {
    namespace kernel {

void fully_connected_fwd_xs_impl(const BTensor & in,
                                 const tensor_t& W,
                                 const tensor_t& b,
                                 BTensor& out,
                                 const params::fully& p,
                                 bool parallelize,
                                 size_t nthreads);

    } // kernel
} // xsdnn

#endif //XSDNN_FULLY_CONNECTED_FWD_XS_IMPL_H
