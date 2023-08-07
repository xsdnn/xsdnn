//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/batch_normalization.h>

namespace xsdnn {

void batch_norm::set_in_shape(const xsdnn::shape3d in_shape) {
    params_.in_shape_ = in_shape;
}

std::vector<shape3d> batch_norm::in_shape() const {
    return {
        params_.in_shape_,
        shape3d(1, 1, params_.in_shape_.D),
        shape3d(1, 1, params_.in_shape_.D)
    };
}

std::vector<shape3d> batch_norm::out_shape() const {
    return { params_.in_shape_ };
}

std::string batch_norm::layer_type() const {
    return "batch_norm";
}

void batch_norm::set_params(mmpack::mm_scalar momentum,
                            mmpack::mm_scalar epsilon, xsdnn::op_mode phase) {
    params_.in_shape_ = shape3d(0, 0, 0);
    params_.momentum_ = momentum;
    params_.eps_ = epsilon;
    params_.phase_ = phase;
}

void batch_norm::init_backend() {
    fwd_kernel_.reset(new core::BatchNormalizationFwdKernel);
    bwd_kernel_.reset(new core::BatchNormalizationBwdKernel);
}

void batch_norm::forward_propagation(const std::vector<tensor_t *> &in_data,
                                     std::vector<tensor_t *> &out_data) {
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.set_engine(engine());
    fwd_ctx_.set_parallelize(parallelize());
    fwd_ctx_.set_num_threads(this->num_threads_);

    fwd_kernel_->compute(fwd_ctx_, params_);
}

} // xsdnn