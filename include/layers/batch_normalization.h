//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BATCH_NORMALIZATION_H
#define XSDNN_BATCH_NORMALIZATION_H

#include "layer.h"
#include "../core/kernel/batch_norm/bn_fwd_kernel.h"

namespace xsdnn {

class batch_norm : public layer {
public:
    explicit batch_norm(mm_scalar momentum = 0.999f, mm_scalar epsilon = 1e-5f,
                        op_mode phase = op_mode::train, core::backend_t engine = core::default_backend_engine())
        : layer({TypeHolder(tensor_type::data, XsDtype::F32),
                 TypeHolder(tensor_type::weight, XsDtype::F32),
                 TypeHolder(tensor_type::bias, XsDtype::F32)},
                {TypeHolder(tensor_type::data, XsDtype::F32)}) {
        set_params(momentum, epsilon, phase);
        init_backend(engine);
    }

public:
    void set_in_shape(const xsdnn::shape3d in_shape);
    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;

    void
    forward_propagation(const std::vector<BTensor *>& in_data,
                        std::vector<BTensor *>& out_data);

    void post_update();

private:
    void set_params(mm_scalar momentum, mm_scalar epsilon, op_mode phase);
    void init_backend(core::backend_t engine);

private:
    params::bnorm params_;
    core::OpContext fwd_ctx_;
    std::shared_ptr<core::BatchNormalizationFwdKernel> fwd_kernel_;
};

} // xsdnn



#endif //XSDNN_BATCH_NORMALIZATION_H
