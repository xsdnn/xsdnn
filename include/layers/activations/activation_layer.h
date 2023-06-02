//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_ACTIVATION_LAYER_H
#define XSDNN_ACTIVATION_LAYER_H

#include "../layer.h"

namespace xsdnn {

class activation_layer : public layer {
public:
    activation_layer();
    activation_layer(const size_t in_size);
    activation_layer(const activation_layer&);
    activation_layer& operator=(const activation_layer&);

public:
    std::vector<shape3d> in_shape() const override;
    std::vector<shape3d> out_shape() const override;
    size_t fan_in_size() const override;
    size_t fan_out_size() const override;

    void set_in_shape(const shape3d in_shape) override;

    void
    forward_propagation(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>& out_data) override;

    void
    back_propagation(const std::vector<tensor_t*>& in_data,
                     const std::vector<tensor_t*>& out_data,
                     std::vector<tensor_t*>&       out_grad,
                     std::vector<tensor_t*>&       in_grad) override;

    virtual
    void
    forward_activation(const mat_t& in_data, mat_t& out_data) = 0;

    virtual
    void
    back_activation(const mat_t& in_data,
                    const mat_t& out_data,
                    const mat_t& out_grad,
                    mat_t&       in_grad) = 0;

    virtual std::pair<mm_scalar, mm_scalar> out_value_range() const override = 0;

    std::string layer_type() const override = 0;

private:
    shape3d in_shape_;
};

} // xsdnn

#endif //XSDNN_ACTIVATION_LAYER_H
