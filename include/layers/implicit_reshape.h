//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_IMPLICIT_RESHAPE_H
#define XSDNN_IMPLICIT_RESHAPE_H

namespace xsdnn {

/*
 * Experimental // ToDO: Impl this or remove
 */
class implicit_reshape {
public:
    explicit implicit_reshape(shape3d& new_shape) : new_shape_(new_shape) {}

public:
    template<typename T>
    T& operator()(T& layer) {

    }

private:
    shape3d new_shape_;
};

} // xsdnn

#endif //XSDNN_IMPLICIT_RESHAPE_H
