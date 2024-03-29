//
// Created by Andrei R. on 28.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_UTILS_H
#define XSDNN_UTILS_H

#define DNN_UNUSED_PARAMETER(x) (void)(x)

namespace xsdnn {

template <typename T>
class index3d {
public:
    index3d(T channel, T width, T height) : channel_(channel), width_(width), height_(height) {}
    index3d() : channel_(0), width_(0), height_(0) {}

    std::array<Index, 3> shape() const {
        std::array<Index, 3> dim = {static_cast<Index>(channel_),
                                    static_cast<Index>(width_),
                                    static_cast<Index>(height_)};
        return dim;
    }

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

using shape3d = index3d<Index>;

template<typename T>
class index4d {
public:
    index4d(T num, T channel, T height, T width) : num_(num), channel_(channel), height_(height), width_(width) {}
    index4d() : num_(0), channel_(0), height_(0), width_(0) {}

    std::array<Index, 4> shape() const {
        std::array<Index, 4> dim = {
                static_cast<Index>(num_),
                static_cast<Index>(channel_),
                static_cast<Index>(height_),
                static_cast<Index>(width_),
        };
        return dim;
    }

    T area() const {
        return (T) height_ * width_;
    }

    T size() const {
        return (T) num_ * channel_ * height_ * width_;
    }

private:
    T num_;
    T channel_;
    T height_;
    T width_;
};

using shape4d = index4d<Index>;

enum class tensor_type : int32_t {
    // input/output data
    data = 0x0001000,  // input/output data

    // trainable parameters
    weight = 0x0002000,
    bias   = 0x0002001,

    label = 0x0004000,
};

bool is_trainable_concept(tensor_type type_) {
    bool value;
    switch (type_) {
        case tensor_type::data:
            value = false;
            break;
        case tensor_type::label:
            value = false;
            break;
        case tensor_type::weight:
            value = true;
            break;
        case tensor_type::bias:
            value = true;
            break;
        default:
            throw xs_error("Unsupported tensor type");
    }
    return value;
}


// FIXME: can we do better at this place?
Matrix transpose(Matrix& m) {
    auto shuffle_idx = std::array<Index, 2> {1, 0};
    return m.shuffle(shuffle_idx);
}

} // xsdnn



#endif //XSDNN_UTILS_H
