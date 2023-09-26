//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/max_pooling.h>

namespace xsdnn {

    std::vector<shape3d> max_pooling::in_shape() const {
        return { params_.in_shape_ };
    }

    std::vector<shape3d> max_pooling::out_shape() const {
        return { params_.out_shape_ };
    }

    void max_pooling::set_params(size_t channels, size_t height, size_t width, size_t kernel_x, size_t kernel_y,
                                 size_t stride_x, size_t stride_y, xsdnn::padding_mode pad_type, bool ceil) {
        params_.in_shape_ = shape3d(channels, height, width);
        size_t h_out = calc_pool_shape(height, kernel_y, stride_y, pad_type, ceil);
        size_t w_out = calc_pool_shape(width, kernel_x, stride_x, pad_type, ceil);
        params_.out_shape_ = shape3d(channels, h_out, w_out);

        params_.kernel_y_ = kernel_y;
        params_.kernel_x_ = kernel_x;
        params_.stride_y_ = stride_y;
        params_.stride_x_ = stride_x;
        params_.pad_type_ = pad_type;
    }

    void max_pooling::init_backend(core::backend_t engine) {
        fwd_kernel_.reset(new core::MaxPoolingFwdKernel);
        connect_inout_idx_loop();
        set_backend(engine);
    }

    std::string max_pooling::layer_type() const {
        return "max_pooling";
    }

    void max_pooling::connect_inout_idx(size_t c, size_t y, size_t x) {
        size_t y_offset_max = std::min(params_.kernel_y_, params_.in_shape_.H - y * params_.stride_y_);
        size_t x_offset_max = std::min(params_.kernel_x_, params_.in_shape_.W - x * params_.stride_x_);

        for (size_t dy = 0; dy < y_offset_max; ++dy) {
            for (size_t dx = 0; dx < x_offset_max; ++dx) {
                size_t in_idx = params_.in_shape_(c, y * params_.stride_y_ + dy, x * params_.stride_x_ + dx);
                size_t out_idx = params_.out_shape_(c, y, x);

                assert(in_idx < params_.in_shape_.size());
                assert(out_idx < params_.out_shape_.size());

                params_.in2out[in_idx] = out_idx;
                params_.out2in[out_idx].push_back(in_idx);
            }
        }
    }

    void max_pooling::connect_inout_idx_loop() {
        params_.in2out.resize(params_.in_shape_.size());
        params_.out2in.resize(params_.out_shape_.size());

        for (size_t c = 0; c < params_.out_shape_.C; ++c) {
            for (size_t h = 0; h < params_.out_shape_.H; ++h) {
                for (size_t w = 0; w < params_.out_shape_.W; ++w) {
                    connect_inout_idx(c, h, w);
                }
            }
        }

    }

    void max_pooling::forward_propagation(const std::vector<BTensor*> &in_data, std::vector<BTensor*> &out_data) {
        fwd_ctx_.set_in_out(in_data, out_data);
        fwd_ctx_.set_engine(this->engine());
        fwd_ctx_.set_parallelize(this->parallelize());
        fwd_ctx_.set_num_threads(this->num_threads_);

        fwd_kernel_->compute(fwd_ctx_, params_);
    }

} // xsdnn