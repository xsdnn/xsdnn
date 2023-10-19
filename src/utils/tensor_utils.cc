//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/tensor_utils.h>
namespace xsdnn {
    namespace tensorize {

void fill(xsDtype dtype, mat_t* p_, size_t size, mm_scalar val) {
    if (dtype == kXsFloat32) {
        gsl::span<float> p_span = GetMutableDataAsSpan<float>(p_);
    }
    for (size_t i = 0; i < size; ++i) {
        *p_ = val;
        p_ += 1;
    }
}

void fill(tensor_t& t_, mm_scalar val) {
    for (auto& m : t_) {
        fill(m.data(), m.size(), val);
    }
}


    } // tensorize
} // xsdnn