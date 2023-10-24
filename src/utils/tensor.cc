//
// Created by rozhin on 24.10.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//
#include <utils/tensor.h>

namespace xsdnn {

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

char* GetMutableDataRaw(mat_t* data, ptrdiff_t byte_offset) {
    return data->data() + byte_offset;
}

const char* GetDataRaw(mat_t* data, ptrdiff_t byte_offset) {
    return data->data() + byte_offset;
}

} // xsdnn