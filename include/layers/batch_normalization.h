//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BATCH_NORMALIZATION_H
#define XSDNN_BATCH_NORMALIZATION_H

#include "layer.h"
#include "../core/kernel/batch_norm/bn_fwd_kernel.h"
#include "../core/kernel/batch_norm/bn_bwd_kernel.h"

namespace xsdnn {

class batch_norm : public layer {
public:
    explicit batch_norm(mm_scalar momentum = 0.999f, mm_scalar epsilon = 1e-5f,
                        op_mode phase = op_mode::train, core::backend_t engine = core::default_backend_engine())
        : layer({tensor_type::data, tensor_type::weight, tensor_type::bias},
                {tensor_type::data}) {
        set_params(momentum, epsilon, phase);
        init_backend(engine);
    }

public:
    void set_in_shape(const xsdnn::shape3d in_shape);
    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;

    void
    forward_propagation(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>& out_data);

    void
    back_propagation(const std::vector<tensor_t*>& in_data,
                     const std::vector<tensor_t*>& out_data,
                     std::vector<tensor_t*>&       out_grad,
                     std::vector<tensor_t*>&       in_grad);

    void post_update();

private:
    void set_params(mm_scalar momentum, mm_scalar epsilon, op_mode phase);
    void init_backend(core::backend_t engine);

private:
    params::bnorm params_;
    core::OpContext fwd_ctx_;
    core::OpContext bwd_ctx_;
    std::shared_ptr<core::BatchNormalizationFwdKernel> fwd_kernel_;
    std::shared_ptr<core::BatchNormalizationBwdKernel> bwd_kernel_;
};

} // xsdnn



#endif //XSDNN_BATCH_NORMALIZATION_H
