//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef MMPACK_TEST_UTILS_H
#define MMPACK_TEST_UTILS_H
#include "../xsdnn.h"
using namespace mmpack;

namespace utils {

void init(mm_scalar* ptr, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            *ptr = i * cols + j;
            ptr += 1;
        }
    }
}

} // utils

#endif //MMPACK_TEST_UTILS_H
