//
// Created by rozhin on 19.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_ADD_H
#define XSDNN_ADD_H

#include "layer.h"

namespace xsdnn {

class add : public layer {
public:
    /*
     * n_input: кол-во входных векторов
     * dim: размерность каждого вектора
     */
    explicit add(size_t n_input, size_t dim)
        : layer(std::vector<TypeHolder>(n_input, TypeHolder(tensor_type::data, XsDtype::F32)),
                {TypeHolder(tensor_type::data, XsDtype::F32)}),
        n_input_(n_input), shape_(1, 1, dim) {}

    explicit add(size_t n_input, shape3d shape)
        : layer(std::vector<TypeHolder>(n_input, TypeHolder(tensor_type::data, XsDtype::F32)),
                {TypeHolder(tensor_type::data, XsDtype::F32)}),
          n_input_(n_input), shape_(shape) {}

    /*
     * Binary add operator
     */
    explicit add(size_t dim)
        : layer(std::vector<TypeHolder>(2, TypeHolder(tensor_type::data, XsDtype::F32)),
                {TypeHolder(tensor_type::data, XsDtype::F32)}),
        n_input_(2), shape_(1, 1, dim) {}

    explicit add(shape3d shape)
        : layer(std::vector<TypeHolder>(2, TypeHolder(tensor_type::data, XsDtype::F32)),
                {TypeHolder(tensor_type::data, XsDtype::F32)}),
          n_input_(2), shape_(shape) {}

    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;

    void
    forward_propagation(const std::vector<BTensor*>& in_data,
                        std::vector<BTensor*>& out_data);

private:
    size_t n_input_;
    shape3d shape_;
    friend struct cerial;
};

} // xsdnn

#endif //XSDNN_ADD_H
