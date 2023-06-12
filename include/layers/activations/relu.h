//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_RELU_H
#define XSDNN_RELU_H

#include <layers/activations/activation_layer.h>

namespace xsdnn {

class relu : public activation_layer {
public:
    using activation_layer::activation_layer;

public:
    void forward_activation(const mat_t& in_data, mat_t& out_data) override;

    void back_activation(const mat_t& in_data,
                         const mat_t& out_data,
                         const mat_t& out_grad,
                         mat_t&       in_grad) override;

    std::pair<mm_scalar, mm_scalar> out_value_range() const override;

    std::string layer_type() const override;
};

} // xsdnn

#endif //XSDNN_RELU_H
