//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef MMPACK_TEST_UTILS_H
#define MMPACK_TEST_UTILS_H
#include "../xsdnn.h"
using namespace mmpack;
using namespace xsdnn;

namespace utils {

void init(mm_scalar* ptr, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            *ptr = i * cols + j;
            ptr += 1;
        }
    }
}

void value_init(mm_scalar* ptr, mm_scalar value, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        *ptr = value;
        ptr += 1;
    }
}

void random_init(mm_scalar* ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        mm_scalar value = static_cast <mm_scalar> (rand()) / static_cast <mm_scalar> (RAND_MAX);
        *ptr = value;
        ptr += 1;
    }
}

std::vector<tensor_t> generate_fwd_data(const size_t num_concept,
                                               const std::vector<size_t> sizes) {
    std::vector<tensor_t> data;
    data.resize(num_concept);
    for (size_t i = 0; i < num_concept; ++i) {
        data[i].resize(1);
        data[i][0].resize(sizes[i]);
        uniform_rand(&data[i][0][0], sizes[i], -10.0f, 10.0f);
    }
    return data;
}

} // utils

#endif //MMPACK_TEST_UTILS_H
