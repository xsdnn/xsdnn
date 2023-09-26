//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/framework/tensor_shape.h>
#include <utils/xs_error.h>

namespace xsdnn {

tensor_shape::tensor_shape(gsl::span<const size_t> values) {
    AllocTmpBuffer(values.size());
    std::copy(values.begin(), values.end(), values_.begin());
}

tensor_shape &tensor_shape::operator=(const xsdnn::tensor_shape &other) {
    if (&other == this)
        return *this;

    AllocTmpBuffer(other.values_.size());
    std::copy(other.get_dims().begin(), other.get_dims().end(), values_.begin());
    return *this;
}

size_t tensor_shape::size() const noexcept {
    size_t size = 1;
    for (size_t s : values_) {
        size *= s;
    }
    return size;
}

tensor_shape tensor_shape::slice(size_t dimstart, size_t dimend) const {
    if (dimstart > dimend || dimend > values_.size()) {
        xsdnn::xs_error("Invalid tensor shape slice argument.");
    }
    return tensor_shape(get_dims().subspan(dimstart, dimend - dimstart));
}

void tensor_shape::AllocTmpBuffer(size_t size) {
    if (values_.size() == size) return;
    tmp_buffer_.reset();
    if (size > kTensorShapeSmallTmpBufferSize) {
        tmp_buffer_ = std::make_unique<size_t>(size);
        values_ = gsl::span<size_t>(tmp_buffer_.get(), size);
    } else {
        values_ = gsl::span<size_t>(small_tmp_buffer, size);
    }
}

}