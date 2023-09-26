//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/tensor_utils.h>

namespace xsdnn {
    namespace tensorize {

void fill(mm_scalar* p_, size_t size, mm_scalar val) {
    for (size_t i = 0; i < size; ++i) {
        *p_ = val;
        p_ += 1;
    }
}

void fill(BTensor& t_, mm_scalar val) {

}


    } // tensorize
} // xsdnn