//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FULLY_CONNECTED_H
#define XSDNN_FULLY_CONNECTED_H

#include "layer.h"
#include "../core/framework/op_context.h"
#include "../core/framework/params.h"
#include "../core/kernel/linear/fully_connected_fwd_kernel.h"
#include "../core/kernel/linear/fully_connected_bwd_kernel.h"

namespace xsdnn {

class fully_connected : public layer {
public:
    fully_connected(size_t in_size,
                    size_t out_size,
                    bool has_bias = true,
                    core::backend_t engine = core::default_backend_engine(),
                    XsDtype dtype = XsDtype::F32)
            :
            layer(get_typed_holder(has_bias, dtype), {TypeHolder(tensor_type::data, dtype)}) {
        set_params(in_size, out_size, has_bias);
        init_backend(engine);
        layer::set_backend(engine);
    }

    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;
    size_t fan_in_size() const;
    size_t fan_out_size() const;

    void
    forward_propagation(const std::vector<BTensor*>& in_data,
                        std::vector<BTensor*>& out_data);

private:
    void set_params(size_t in_size, size_t out_size, bool has_bias);
    void init_backend(core::backend_t engine);

private:
    params::fully params_;
    core::OpContext fwd_ctx_;
    core::OpContext bwd_ctx_;
    std::shared_ptr<core::FullyConnectedFwdKernel> fwd_kernel_;

    friend struct cerial;
};

} // xsdnn

#endif //XSDNN_FULLY_CONNECTED_H
