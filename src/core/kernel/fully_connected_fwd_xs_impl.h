//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FULLY_CONNECTED_FWD_XS_IMPL_H
#define XSDNN_FULLY_CONNECTED_FWD_XS_IMPL_H

#include "../framework/params.h"
#include "../../utils/tensor.h"

namespace xsdnn {
    namespace kernel {

inline void fully_connected_fwd_xs_impl(const tensor_t& in,
                                        const mat_t& W,
                                        const mat_t& b,
                                        tensor_t& out,
                                        const params::fully& p,
                                        bool parallelize);

    } // kernel
} // xsdnn

#endif //XSDNN_FULLY_CONNECTED_FWD_XS_IMPL_H
