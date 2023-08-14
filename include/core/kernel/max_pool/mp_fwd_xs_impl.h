//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MP_FWD_XS_IMPL_H
#define XSDNN_MP_FWD_XS_IMPL_H

#include "../../framework/params.h"
#include "../../../utils/tensor.h"

namespace xsdnn {
    namespace kernel {

        void max_pool_fwd_xs_impl(const tensor_t& in,
                                  tensor_t& out,
                                  params::max_pool& p,
                                  bool parallelize,
                                  size_t nthreads);

    }
}

#endif //XSDNN_MP_FWD_XS_IMPL_H
