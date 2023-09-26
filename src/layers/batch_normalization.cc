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
        shape3d(params_.in_shape_.C, 1, 1),
        shape3d(params_.in_shape_.C, 1, 1)
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

void batch_norm::init_backend(core::backend_t engine) {
    fwd_kernel_.reset(new core::BatchNormalizationFwdKernel);
    this->weight_init(weight_init::constant(1.0f));
    this->bias_init(weight_init::constant(0.0f));
    set_backend(engine);
}

void batch_norm::forward_propagation(const std::vector<BTensor*> &in_data,
                                     std::vector<BTensor*> &out_data) {
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.set_engine(engine());
    fwd_ctx_.set_parallelize(parallelize());
    fwd_ctx_.set_num_threads(this->num_threads_);

    fwd_kernel_->compute(fwd_ctx_, params_);
}

void batch_norm::post_update() {
    gsl::span<float> mean_ = params_.stat_holder["mean_"].GetMutableDataAsSpan<float>();
    gsl::span<float> stddev_ = params_.stat_holder["stddev_"].GetMutableDataAsSpan<float>();
    gsl::span<const float> mean_running_ = params_.stat_holder["mean_running_"].GetDataAsSpan<float>();
    gsl::span<const float> stddev_running_ = params_.stat_holder["stddev_running_"].GetDataAsSpan<float>();
    float momentum_ = params_.momentum_;

    for (size_t i = 0; i < mean_.size(); i++) {
        mean_[i] = momentum_ * mean_[i] + (1 - momentum_) * mean_running_[i];
        stddev_[i] = momentum_ * stddev_[i] + (1 - momentum_) * stddev_running_[i];
    }
}

} // xsdnn