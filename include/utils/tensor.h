//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_H
#define XSDNN_TENSOR_H

#include <vector>
#include "../mmpack/mmpack.h"
#include "../utils/xs_error.h"
#include "../gsl/span"

using namespace mmpack;

namespace xsdnn {
    typedef std::vector<char, aligned_allocator<char, 64>> mat_t;
    typedef std::vector<mat_t> tensor_t;

    typedef enum {
        kXsUndefined = 0,
        kXsFloat32 = 1,
        kXsFloat16 = 2
    } xsDtype;

    size_t dtype2sizeof(xsDtype dtype) {
        switch(dtype) {
            case kXsFloat32:
                return sizeof(float);
            case kXsFloat16:
                return sizeof(uint16_t);
            default:
                throw xs_error("Unsupported dtype when converting to sizeof(...)");
        }
    }

    void AllocateMat_t(mat_t* data, size_t size, xsDtype dtype) {
        data->resize(size * dtype2sizeof(dtype));
    }

    template<typename T>
    gsl::span<T> GetMutableDataAsSpan(mat_t* data, ptrdiff_t byte_offset = 0) {
        T* ptr = reinterpret_cast<T*>(data->data() + byte_offset);
        return gsl::make_span(ptr, data->size());
    }

    template<typename T>
    gsl::span<const T> GetDataAsSpan(mat_t* data, ptrdiff_t byte_offset = 0) {
        T* ptr = reinterpret_cast<T*>(data->data() + byte_offset);
        return gsl::make_span(ptr, data->size());
    }

    char* GetMutableDataRaw(mat_t* data, ptrdiff_t byte_offset = 0) {
        return data->data() + byte_offset;
    }

    const char* GetDataRaw(mat_t* data, ptrdiff_t byte_offset = 0) {
        return data->data() + byte_offset;
    }
}


#endif //XSDNN_TENSOR_H
