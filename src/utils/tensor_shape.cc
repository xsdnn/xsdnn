//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/tensor_shape.h>

namespace xsdnn {

shape3d::shape3d(size_t height, size_t width, size_t depth) {
    reshape(height, width, depth);
}

shape3d::shape3d() {
    H = 0; W = 0; C = 0;
}

void shape3d::reshape(size_t height, size_t width, size_t depth) {
    H = height;
    W = width;
    C = depth;
}

size_t shape3d::operator()(size_t y, size_t x, size_t channel) {
    assert(y >= 0 && y < H);
    assert(x >= 0 && x < W);
    assert(channel >= 0 && channel < C);
    return (C * channel + y) * W + x;
}

bool shape3d::operator==(const shape3d &rhs) {
    return (H == rhs.H) && (W == rhs.W) && (C == rhs.C);
}

bool shape3d::operator!=(const shape3d &rhs) {
    return !(*this == rhs);
}

size_t shape3d::area() const { return (size_t) H * W; }
size_t shape3d::size() const { return (size_t) H * W * C; }

std::ostream& operator<<(std::ostream& out, const shape3d& obj) {
    out << "[" << obj.H << ", " << obj.W << ", " << obj.C << "]";
    return out;
}

std::ostream& operator<<(std::ostream& out, const std::vector<shape3d>& obj) {
    out << obj[0];
    return out;
}

}






