//
// Created by rozhin on 27.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_AND_H
#define XSDNN_AND_H

#include "layer.h"

namespace xsdnn {

class and_layer : public layer {
public:
    explicit and_layer()
            : layer({tensor_type::data, tensor_type::data}, {tensor_type::data}),
              shape_() {}

    explicit and_layer(shape3d shape)
    : layer({tensor_type::data, tensor_type::data}, {tensor_type::data}),
    shape_(shape) {}

    explicit and_layer(size_t dim)
    : layer({tensor_type::data, tensor_type::data}, {tensor_type::data}),
    shape_(1, dim, 1) {}

public:
    void set_in_shape(const xsdnn::shape3d in_shape);
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
    bool contains_only_one_zero(const tensor_t& mat);

private:
    shape3d shape_;
};

} // xsdnn

#endif //XSDNN_AND_H
