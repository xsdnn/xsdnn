//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_SHAPE_H
#define XSDNN_TENSOR_SHAPE_H

#include "tensor.h"
#include <cassert>
#include <iostream>

namespace xsdnn {

struct shape3d {
public:
    shape3d(size_t channel, size_t height, size_t width);
    shape3d();

    size_t operator() (size_t channel, size_t y, size_t x);
    bool operator == (const shape3d& rhs);
    bool operator != (const shape3d& rhs);

    void reshape(size_t channel, size_t width, size_t height);
    std::vector<size_t> get_dims() const noexcept;

    size_t area() const;
    size_t size() const;

public:
    size_t C;
    size_t H;
    size_t W;
};

std::ostream& operator<<(std::ostream& out, const shape3d& obj);
std::ostream& operator<<(std::ostream& out, const std::vector<shape3d>& obj);

} // xsdnn



#endif //XSDNN_TENSOR_SHAPE_H
