//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/fully_connected.h>

namespace xsdnn {

std::vector<shape3d> fully_connected::in_shape() const {
    if (params_.has_bias_) {
        return {
            shape3d(1, 1, params_.in_size_),
            shape3d(1, params_.in_size_, params_.out_size_),
            shape3d(1, 1, params_.out_size_)
        };
    } else {
        return {
                shape3d(1, 1, params_.in_size_),
                shape3d(1, params_.in_size_, params_.out_size_),
        };
    }
}

std::vector<shape3d> fully_connected::out_shape() const {
    return {
        shape3d(1, 1, params_.out_size_)
    };
}

std::string fully_connected::layer_type() const {
    return "fully_connected";
}

size_t fully_connected::fan_in_size() const {
    return params_.in_size_;
}

size_t fully_connected::fan_out_size() const {
    return params_.out_size_;
}

void fully_connected::forward_propagation(
        const std::vector<tensor_t *> &in_data,
        std::vector<tensor_t *> &out_data) {

    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.set_engine(layer::engine());
    fwd_ctx_.set_parallelize(layer::parallelize());
    fwd_ctx_.set_num_threads(layer::num_threads_);

    fwd_kernel_->compute(fwd_ctx_, params_);
}

void fully_connected::set_params(size_t in_size,
                                 size_t out_size,
                                 bool has_bias) {
    params_.in_size_ = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
}

void fully_connected::init_backend(core::backend_t engine) {
    fwd_kernel_.reset(new core::FullyConnectedFwdKernel);
}

} // xsdnn