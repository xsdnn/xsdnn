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
        for (float& value : p_span) {
            value = val;
        }
    } else throw xs_error("Unsupported dtype at weight_init");
}


    } // tensorize
} // xsdnn