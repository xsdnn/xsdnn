//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/batch_norm/bn_bwd_xs_impl.h>

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
                                         size_t nthreads) {
        // TODO: make it
        throw xs_error("[batch_norm backward xs engine] Not Implemented Yet");
    }

    } // kernel
} // xsdnn