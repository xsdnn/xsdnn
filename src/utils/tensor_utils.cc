//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/tensor_utils.h>
namespace xsdnn {
    namespace tensorize {

void fill(xsDtype dtype, mat_t* p_, float val) {
    if (dtype == kXsFloat32) {
        gsl::span<float> p_span = GetMutableDataAsSpan<float>(p_);
        for (size_t i = 0; i < p_span.size() / sizeof(float); ++i) {
            p_span[i] = val;
        }
    } else throw xs_error("Unsupported dtype at weight_init");
}


    } // tensorize
} // xsdnn