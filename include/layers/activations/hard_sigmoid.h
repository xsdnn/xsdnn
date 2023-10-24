//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_HARD_SIGMOID_H
#define XSDNN_HARD_SIGMOID_H

#include "activation_layer.h"

namespace xsdnn {

class hard_sigmoid : public activation_layer {
public:
    explicit hard_sigmoid(const size_t in_size,
                          const float alpha = 0.2f,
                          const float beta = 0.5f)
            : activation_layer(in_size), alpha_(alpha), beta_(beta) {}

    explicit hard_sigmoid(const float alpha = 0.2f,
                          const float beta = 0.5f)
        : activation_layer(), alpha_(alpha), beta_(beta) {}

public:
    std::pair<mm_scalar, mm_scalar> out_value_range() const override;
    std::string layer_type() const override;

protected:
    virtual void init_params() override;

private:
    float alpha_;
    float beta_;
    friend struct cerial;
};

} // xsdnn

#endif //XSDNN_HARD_SIGMOID_H
