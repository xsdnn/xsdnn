//
// Created by rozhin on 04.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_FLATTEN_H
#define XSDNN_FLATTEN_H

#include "layer.h"

namespace xsdnn {

class flatten : public layer {
public:
    explicit flatten()
        : layer({tensor_type::data}, {tensor_type::data}),
          in_shape_() {}

    explicit flatten(shape3d shape)
        : layer({tensor_type::data}, {tensor_type::data}),
          in_shape_(shape) {}

    explicit flatten(size_t dim)
        : layer({tensor_type::data}, {tensor_type::data}),
          in_shape_(1, 1, dim) {}

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
    shape3d in_shape_;
};

} // xsdnn

#endif //XSDNN_FLATTEN_H
