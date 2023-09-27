//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_H
#define XSDNN_TENSOR_H

#include "allocator.h"
#include "../../utils/tensor_shape.h"
#include "../../gsl/span"

namespace xsdnn {

static CPUAllocator CPUMalloc;

class tensor_t;
typedef std::vector<tensor_t> BTensor;

// TODO: добавить операторы сравнения, копирования (оператор + конструктор)
// TODO: продумать логику установки тензоров в качестве весов


enum XsDtype {
    UND  = 0,   // UNDEFINED
    F32  = 1,   // FLOAT32
    F16  = 2,   // FLOAT16
    INT8 = 3,   // INT8
    INT4 = 4    // INT4
};

size_t sizeofDtype(XsDtype dtype);

class tensor_t final {
public:
    tensor_t() : shape_(shape3d(0, 0, 0)), dtype_(XsDtype::UND), p_data_(nullptr), allocator_(nullptr) {}
    explicit tensor_t(XsDtype p_type, const shape3d& shape, IAllocator* allocator);
    explicit tensor_t(XsDtype p_type, const size_t size, IAllocator* allocator)
        : tensor_t(p_type, shape3d(1, 1, size), allocator) {}
    ~tensor_t() {}

public:
    shape3d shape() const { return shape_; }
    XsDtype dtype() const { return dtype_; }

    bool empty() const { return shape_.size() == 0; }

    template<typename T>
    T* GetMutableData(ptrdiff_t byte_offset = 0) const {
        return reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
    }

    template<typename T>
    gsl::span<T> GetMutableDataAsSpan(ptrdiff_t byte_offset = 0) const {
        T* data = reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
        return gsl::make_span(data, shape_.size());
    }

    template<typename T>
    const T* GetData(ptrdiff_t byte_offset = 0) const {
        return reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
    }

    template<typename T>
    gsl::span<const T> GetDataAsSpan(ptrdiff_t byte_offset = 0) const {
        T* data = reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset);
        return gsl::make_span(data, shape_.size());
    }

    const void* GetDataRaw(ptrdiff_t byte_offset = 0) const {
        return static_cast<char*>(p_data_) + byte_offset;
    }

    void* GetMutableDataRaw(ptrdiff_t byte_offset = 0) const {
        return static_cast<char*>(p_data_) + byte_offset;
    }

private:
    void Init(XsDtype p_type, const shape3d& shape, IAllocator* allocator);
    void AllocTensor();
    void FreeTensor();

private:
    shape3d shape_;
    XsDtype dtype_;
    void* p_data_;

    IAllocator* allocator_;
};

}

#endif //XSDNN_TENSOR_H
