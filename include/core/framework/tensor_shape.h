//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_SHAPE_H
#define XSDNN_TENSOR_SHAPE_H

#include "../../gsl/span"

namespace xsdnn {

class tensor_shape {
public:
    explicit tensor_shape() {}
    explicit tensor_shape(gsl::span<const size_t> values);
    explicit tensor_shape(std::initializer_list<size_t> values) : tensor_shape(gsl::make_span(values.begin(), values.end())) {}
    tensor_shape(tensor_shape& other) : tensor_shape(other.get_dims()) {}
    tensor_shape& operator=(const tensor_shape& other);

    size_t operator[](size_t idx) const { return values_[idx]; }
    size_t& operator[](size_t idx) { return values_[idx]; }

    bool operator==(tensor_shape& other) { return other.get_dims() == values_;}
    bool operator!=(tensor_shape& other) { return !(other == *this); }

    gsl::span<const size_t> get_dims() const { return values_; }
    gsl::span<size_t> get_dims() { return values_; }

    /*
     * Ранг тензора
     */
    size_t num_dimensions() const noexcept { return values_.size(); }

    /*
     * Prod(dim_1, ..., dim_n)
     */
    size_t size() const noexcept;

    tensor_shape slice(size_t dimstart, size_t dimend) const;
    tensor_shape slice(size_t dimstart) const { return slice(dimstart, values_.size()); }

private:
    void AllocTmpBuffer(size_t size);

private:
    static constexpr size_t kTensorShapeSmallTmpBufferSize = 4;
    gsl::span<size_t> values_;
    size_t small_tmp_buffer[kTensorShapeSmallTmpBufferSize]{0};
    std::unique_ptr<size_t> tmp_buffer_;
};

}

#endif //XSDNN_TENSOR_SHAPE_H
