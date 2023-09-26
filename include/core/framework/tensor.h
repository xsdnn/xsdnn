//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_H
#define XSDNN_TENSOR_H

#include "allocator.h"
#include "tensor_shape.h"

namespace xsdnn {

enum XsDtype {
    F32  = 0,
    F16  = 1,
    Int8 = 2,
    Int4 = 3
};

class tensor_t final {
public:
    explicit tensor_t(XsDtype p_type, const tensor_shape& shape, IAllocator* allocator);
    explicit tensor_t(XsDtype p_type, const size_t size, IAllocator* allocator)
        : tensor_t(p_type, tensor_shape({size}), allocator) {}
    ~tensor_t();

public:
    tensor_shape shape() { return shape_; }
    XsDtype dtype() { return dtype_; }

    template<typename T>
    T* GetMutableData(ptrdiff_t byte_offset = 0) {
        return reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
    }

    template<typename T>
    gsl::span<T> GetMutableDataAsSpan(ptrdiff_t byte_offset = 0) {
        T* data = reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
        return gsl::make_span(data, shape_.size());
    }

    template<typename T>
    const T* GetData(ptrdiff_t byte_offset = 0) {
        return reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
    }

    template<typename T>
    gsl::span<const T> GetDataAsSpan(ptrdiff_t byte_offset = 0) {
        T* data = reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
        return gsl::make_span(data, shape_.size());
    }

private:
    void Init(XsDtype p_type, const tensor_shape& shape, IAllocator* allocator);
    void AllocTensor();
    void FreeTensor();

private:
    tensor_shape shape_;
    XsDtype dtype_;
    void* p_data_;

    IAllocator* allocator_;
};

}

#endif //XSDNN_TENSOR_H
