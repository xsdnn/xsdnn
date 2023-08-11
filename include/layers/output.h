//
// Created by rozhin on 09.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_OUTPUT_H
#define XSDNN_OUTPUT_H

#include "layer.h"

namespace xsdnn {

class Output : public layer {
public:
    explicit Output()
    : layer({tensor_type::data}, {tensor_type::data}),
    shape_(shape3d(0, 0, 0)) {}

    explicit Output(shape3d shape)
    : layer({tensor_type::data}, {tensor_type::data}),
    shape_(shape) {}

    explicit Output(size_t in_size)
    : layer({tensor_type::data}, {tensor_type::data}),
    shape_(1, in_size, 1) {}

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
    shape3d shape_;
    friend struct cerial;
};

} // xsdnn

#endif //XSDNN_OUTPUT_H
