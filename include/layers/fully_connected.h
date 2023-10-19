//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FULLY_CONNECTED_H
#define XSDNN_FULLY_CONNECTED_H

#include "layer.h"
#include "../core/framework/op_context.h"
#include "../core/framework/params.h"
#include "../core/kernel/fully_connected_fwd.h"

namespace xsdnn {

class fully_connected : public layer {
public:
    fully_connected(size_t in_size,
                    size_t out_size,
                    bool has_bias = true,
                    core::backend_t engine = core::default_backend_engine(),
                    xsDtype dtype = kXsFloat32)
            :
            layer(define_input_bias_condition(has_bias), {tensor_type::data}, dtype) {
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
    forward_propagation(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>& out_data);


private:
    void set_params(size_t in_size, size_t out_size, bool has_bias);
    void init_backend(core::backend_t engine);

private:
    params::fully params_;
    core::OpContext fwd_ctx_;
    std::shared_ptr<core::FullyConnectedFwdKernel> fwd_kernel_;

    friend struct cerial;
};

} // xsdnn

#endif //XSDNN_FULLY_CONNECTED_H
