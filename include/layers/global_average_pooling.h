//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_GLOBAL_AVERAGE_POOLING_H
#define XSDNN_GLOBAL_AVERAGE_POOLING_H

#include "layer.h"
#include "../core/kernel/global_average_pooling/gap_fwd_kernel.h"

namespace xsdnn {

class global_average_pooling : public layer {
public:
    explicit global_average_pooling(shape3d shape,
                                    core::backend_t engine = core::default_backend_engine())
            : layer({tensor_type::data}, {tensor_type::data}, xsDtype::kXsFloat32) {
        set_params(shape.C, shape.H, shape.W);
        init_backend(engine);
    }

    explicit global_average_pooling(size_t channels,
                         size_t height,
                         size_t width,
                         core::backend_t engine = core::default_backend_engine())
        : layer({tensor_type::data}, {tensor_type::data}, xsDtype::kXsFloat32) {
        set_params(channels, height, width);
        init_backend(engine);
    }

public:
    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;

    void
    forward_propagation(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>& out_data);

private:
    void set_params(size_t channels,
                    size_t height,
                    size_t width);

    void init_backend(core::backend_t engine);

private:
    params::global_avg_pool params_;
    core::OpContext fwd_ctx_;
    std::shared_ptr<core::GlobalAvgPoolingFwdKernel> fwd_kernel_;
    friend struct cerial;
};

} // xsdnn

#endif //XSDNN_GLOBAL_AVERAGE_POOLING_H
