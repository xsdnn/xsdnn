//
// Created by rozhin on 17.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_RESHAPE_H
#define XSDNN_RESHAPE_H

#include "layer.h"

namespace xsdnn {

class reshape : public layer {
public:
    explicit reshape(shape3d in_shape)
        : reshape(in_shape.C, in_shape.H, in_shape.W) {}

    explicit reshape(size_t channel,
                     size_t height,
                     size_t width)
        : layer({TypeHolder(tensor_type::data, XsDtype::F32)}, {TypeHolder(tensor_type::data, XsDtype::F32)}),
        in_shape_(0, 0, 0), out_shape_(channel, height, width) {}

public:
    void set_in_shape(const xsdnn::shape3d in_shape);
    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;

    void
    forward_propagation(const std::vector<BTensor*>& in_data,
                        std::vector<BTensor*>& out_data);

private:
    shape3d in_shape_;
    shape3d out_shape_;
    friend struct cerial;
};

} // xsdnn

#endif //XSDNN_RESHAPE_H
