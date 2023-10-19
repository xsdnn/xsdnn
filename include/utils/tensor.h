//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_H
#define XSDNN_TENSOR_H

#include <vector>
#include "../mmpack/mmpack.h"
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

    template<typename T>
    gsl::span<T> GetMutableDataAsSpan(mat_t* data) {
        return gsl::span<T>(reinterpret_cast<T*>(data->data(), data->size()));
    }

    template<typename T>
    gsl::span<const T> GetDataAsSpan(mat_t* data) {
        return gsl::span<const T>(reinterpret_cast<T*>(data->data(), data->size()));
    }

    char* GetMutableDataRaw(mat_t* data) {
        return data->data();
    }

    const char* GetDataRaw(mat_t* data) {
        return data->data();
    }
}


#endif //XSDNN_TENSOR_H
