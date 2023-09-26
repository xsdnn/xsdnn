//
// Created by rozhin on 14.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MUL_H
#define XSDNN_MUL_H

#include "layer.h"

namespace xsdnn {

class mul : public layer {
    explicit mul()
        : layer({TypeHolder(tensor_type::data, XsDtype::F32),
                 TypeHolder(tensor_type::data, XsDtype::F32)},
                {TypeHolder(tensor_type::data, XsDtype::F32)}), in_shape_() {}

    explicit mul(size_t dim)
        : layer({TypeHolder(tensor_type::data, XsDtype::F32),
                 TypeHolder(tensor_type::data, XsDtype::F32)},
                {TypeHolder(tensor_type::data, XsDtype::F32)}), in_shape_(1, 1, dim) {}

    explicit mul(shape3d in_shape)
        : layer({TypeHolder(tensor_type::data, XsDtype::F32),
                 TypeHolder(tensor_type::data, XsDtype::F32)},
                {TypeHolder(tensor_type::data, XsDtype::F32)}), in_shape_(in_shape) {}

public:
    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;

    void
    forward_propagation(const std::vector<BTensor*>& in_data,
                        std::vector<BTensor*>& out_data);

private:
    shape3d in_shape_;
};

} // xsdnn

#endif //XSDNN_MUL_H
