//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/tensor_shape.h>

namespace xsdnn {

shape3d::shape3d(size_t channel, size_t height, size_t width) {
    reshape(channel, height, width);
}

shape3d::shape3d() {
    H = 0; W = 0; C = 0;
}

void shape3d::reshape(size_t channel, size_t height, size_t width) {
    C = channel;
    H = height;
    W = width;
}

std::vector<size_t> shape3d::get_dims() const noexcept {
    return {C, H, W};
}

size_t shape3d::operator()(size_t channel, size_t y, size_t x) {
    assert(y >= 0 && y < H);
    assert(x >= 0 && x < W);
    assert(channel >= 0 && channel < C);
    return W * (channel * H + y) + x;
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
    out << "[" << obj.C << ", " << obj.H << ", " << obj.W << "]";
    return out;
}

std::ostream& operator<<(std::ostream& out, const std::vector<shape3d>& obj) {
    out << obj[0];
    return out;
}

}






