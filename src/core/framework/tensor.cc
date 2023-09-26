//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/framework/tensor.h>
#include <utils/xs_error.h>

namespace xsdnn {

tensor_t::tensor_t(XsDtype p_type, const tensor_shape &shape,
                   IAllocator *allocator) {
    Init(p_type, shape, allocator);
}

tensor_t::~tensor_t() {
    FreeTensor();
}

void tensor_t::Init(XsDtype p_type, const tensor_shape &shape,
                    IAllocator *allocator) {
    size_t shape_size = shape.size();
    if (shape_size <= 0)
        xs_error("Tensor Shape size must be greater than zero.");
    dtype_ = p_type;
    shape_ = shape;
    allocator_ = allocator;
    AllocTensor();
}

void tensor_t::AllocTensor() {
    p_data_ = allocator_->Alloc(shape_.size());
}

void tensor_t::FreeTensor() {
    if (p_data_) {
        allocator_->Free(p_data_);
    }
}

} // xsdnn