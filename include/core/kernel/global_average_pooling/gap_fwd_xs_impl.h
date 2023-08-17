//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_GAP_FWD_XS_IMPL_H
#define XSDNN_GAP_FWD_XS_IMPL_H

#include "../../framework/params.h"
#include "../../../utils/tensor.h"

namespace xsdnn {
    namespace kernel {

        void global_average_pool_fwd_xs_impl(const tensor_t& in,
                                  tensor_t& out,
                                  params::global_avg_pool& p,
                                  bool parallelize,
                                  size_t nthreads);

    }
}

#endif //XSDNN_GAP_FWD_XS_IMPL_H
