//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_SHAPE_H
#define XSDNN_TENSOR_SHAPE_H

#include "tensor.h"

struct shape3d {
public:
    shape3d(size_t width, size_t height, size_t depth);
    shape3d();

    size_t operator() (size_t x, size_t y, size_t channel);

    void reshape(size_t width, size_t height, size_t depth);

    size_t area() const;
    size_t size() const;

public:
    size_t W;
    size_t H;
    size_t D;
};

#endif //XSDNN_TENSOR_SHAPE_H
