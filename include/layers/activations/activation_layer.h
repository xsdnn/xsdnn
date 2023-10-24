//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_ACTIVATION_LAYER_H
#define XSDNN_ACTIVATION_LAYER_H

#include "../layer.h"
#include "../../core/kernel/activation_fwd.h"

#define ACTIVATION_INIT_PARAMS_LAST_LINE    \
p_.shape = this->in_shape()[0];             \
initialized_ = true;

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

    virtual std::pair<mm_scalar, mm_scalar> out_value_range() const override = 0;

    std::string layer_type() const override = 0;

protected:
    params::activation p_;
    // Необходимо определить параметры активации
    // в атрибуте p_. В конце метода добавить initialized_ = true;
    virtual void init_params() = 0;
    bool initialized_;

private:
    void init_backend();

private:
    shape3d in_shape_;
    core::OpContext fwd_ctx_;
    std::shared_ptr<core::ActivationFwdKernel> fwd_kernel_;
};

} // xsdnn

#endif //XSDNN_ACTIVATION_LAYER_H
