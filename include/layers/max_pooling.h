//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MAX_POOLING_H
#define XSDNN_MAX_POOLING_H

#include "layer.h"
#include "../core/kernel/max_pool/mp_fwd_kernel.h"

namespace xsdnn {

class max_pooling : public layer {
public:
    explicit max_pooling(size_t channels,
                         size_t height,
                         size_t width,
                         size_t kernel_xy,
                         size_t stride_xy,
                         padding_mode pad_type = padding_mode::valid,
                         bool   ceil = false,
                         core::backend_t engine = core::default_backend_engine())
        : max_pooling(channels, height, width, kernel_xy, kernel_xy, stride_xy, stride_xy, pad_type, ceil, engine) {}

    explicit max_pooling(shape3d in_shape,
                         size_t kernel_xy,
                         size_t stride_xy,
                         padding_mode pad_type = padding_mode::valid,
                         bool   ceil = false,
                         core::backend_t engine = core::default_backend_engine())
        : max_pooling(in_shape.C, in_shape.H, in_shape.W, kernel_xy, kernel_xy, stride_xy, stride_xy, pad_type, ceil, engine) {}

    explicit max_pooling(shape3d in_shape,
                         size_t kernel_x,
                         size_t kernel_y,
                         size_t stride_x,
                         size_t stride_y,
                         padding_mode pad_type = padding_mode::valid,
                         bool   ceil = false,
                         core::backend_t engine = core::default_backend_engine())
        : max_pooling(in_shape.C, in_shape.H, in_shape.W, kernel_x, kernel_y, stride_x, stride_y, pad_type, ceil, engine) {}

    explicit max_pooling(size_t channels,
                         size_t height,
                         size_t width,
                         size_t kernel_x,
                         size_t kernel_y,
                         size_t stride_x,
                         size_t stride_y,
                         padding_mode pad_type = padding_mode::valid,
                         bool   ceil = false,
                         core::backend_t engine = core::default_backend_engine())
        : layer({tensor_type::data}, {tensor_type::data}) {
        set_params(channels, height, width, kernel_x, kernel_y, stride_x, stride_y, pad_type, ceil);
        init_backend(engine);
    }

public:
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

private:
    void set_params(size_t channels,
                    size_t height,
                    size_t width,
                    size_t kernel_x,
                    size_t kernel_y,
                    size_t stride_x,
                    size_t stride_y,
                    padding_mode pad_type,
                    bool   ceil = false);

    void init_backend(core::backend_t engine);

    /*
     * Сопоставление индексов входа и выхода, из которых будет браться максимум
     */
    void connect_inout_idx(size_t c, size_t y, size_t x);
    void connect_inout_idx_loop();

private:
    params::max_pool params_; // TODO: нужен ли in2out???
    core::OpContext fwd_ctx_;
    std::shared_ptr<core::MaxPoolingFwdKernel> fwd_kernel_;
};

} // xsdnn

#endif //XSDNN_MAX_POOLING_H
