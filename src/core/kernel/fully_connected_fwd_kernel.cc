//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "fully_connected_fwd_kernel.h"
#include "fully_connected_fwd_xs_impl.h"
#include "../../utils/tensor_utils.h"
#include "../../utils/xs_error.h"

namespace xsdnn {
    namespace core {

void FullyConnectedFwdKernel::compute(OpContext &ctx, params::fully &p) {
    const tensor_t& in = ctx.input_data(0);
    const tensor_t& W = ctx.input_data(1);
    const tensor_t* b = p.has_bias_ ? &ctx.input_data(2) : nullptr;
    tensor_t& out = ctx.output_data(0);

    tensorize::fill(out, (mm_scalar) 0.0f); // FIXME: Можно ли убрать это?

    core::backend_t engine = ctx.engine();

    if (engine == core::backend_t::xs) {
        kernel::fully_connected_fwd_xs_impl(in,
                                            W[0],
                                            p.has_bias_ ? (*b)[0] : mat_t(),
                                            out, p, ctx.parallelize());
    } else {
        throw xs_error("Unsupported engine type"); // TODO: расширить на понятную ошибку
    }
}

    } // xsdnn
} // xsdnn