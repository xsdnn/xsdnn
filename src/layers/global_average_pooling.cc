//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/global_average_pooling.h>

namespace xsdnn {

    void global_average_pooling::set_params(size_t channels, size_t height, size_t width) {
        params_.in_shape_ = shape3d(channels, height, width);
        params_.out_shape_ = shape3d(channels, 1, 1);
    }

    void global_average_pooling::init_backend(core::backend_t engine) {
        fwd_kernel_.reset(new core::GlobalAvgPoolingFwdKernel);
        set_backend(engine);
    }

    std::vector<shape3d> global_average_pooling::in_shape() const {
        return { params_.in_shape_ };
    }

    std::vector<shape3d> global_average_pooling::out_shape() const {
        return { params_.out_shape_ };
    }

    std::string global_average_pooling::layer_type() const {
        return "global_average_pooling";
    }

    void global_average_pooling::forward_propagation(const std::vector<tensor_t *> &in_data,
                                                     std::vector<tensor_t *> &out_data) {
        fwd_ctx_.set_in_out(in_data, out_data);
        fwd_ctx_.set_parallelize(this->parallelize());
        fwd_ctx_.set_num_threads(this->num_threads_);
        fwd_ctx_.set_engine(this->engine());

        fwd_kernel_->compute(fwd_ctx_, params_);
    }

} // xsdnn
