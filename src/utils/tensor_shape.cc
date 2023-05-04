//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "tensor_shape.h"

namespace xsdnn {

shape3d::shape3d(size_t width, size_t height, size_t depth) {
    reshape(width, height, depth);
}

shape3d::shape3d() {
    W = 0; H = 0; D = 0;
}

void shape3d::reshape(size_t width, size_t height, size_t depth) {
    W = width;
    H = height;
    D = depth;
}

size_t shape3d::operator()(size_t x, size_t y, size_t channel) {
    assert(x >= 0 && x < W);
    assert(y >= 0 && y < H);
    assert(channel >= 0 && channel < D);
    return (D * channel + y) * W + x;
}

bool shape3d::operator==(const shape3d &rhs) {
    return (W == rhs.W) && (H == rhs.H) && (D == rhs.D);
}

bool shape3d::operator!=(const shape3d &rhs) {
    return !(*this == rhs);
}

size_t shape3d::area() const { return (size_t) W * H; }
size_t shape3d::size() const { return (size_t) W * H * D; }

}






