//
// Created by Andrei R. on 28.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_UTILS_H
#define XSDNN_UTILS_H

namespace xsdnn {

template <typename T>
class index3d {
public:
    index3d(T channel, T width, T height) : channel_(channel), width_(width), height_(height) {}
    index3d() : channel_(0), width_(0), height_(0) {}

    T area() const {
        return width_ * height_;
    }

    T size() const {
        return channel_ * width_ * height_;
    }

private:
    T channel_;
    T width_;
    T height_;
};

using shape3d = index3d<size_t>;

enum class tensor_type : int32_t {
    // input/output data
    data = 0x0001000,  // input/output data

    // trainable parameters
    weight = 0x0002000,
    bias   = 0x0002001,

    label = 0x0004000,
};

} // xsdnn



#endif //XSDNN_UTILS_H
