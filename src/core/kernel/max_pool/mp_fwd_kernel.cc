//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/kernel/max_pool/mp_fwd_kernel.h>
#include <core/kernel/max_pool/mp_fwd_xs_impl.h>

namespace xsdnn {
    namespace core {

void MaxPoolingFwdKernel::compute(xsdnn::core::OpContext &ctx, params::max_pool &p) {
    const BTensor& in_data = ctx.input_data(0);
    BTensor& out_data = ctx.output_data(0);

    backend_t engine = ctx.engine();

    if (engine == backend_t::xs) {
        kernel::max_pool_fwd_xs_impl(in_data, out_data, p, ctx.parallelize(), ctx.num_threads());
    } else {
        xs_error("[max_pool forward] unsupported engine type");
    }
}

    } // core
} // xsdnn