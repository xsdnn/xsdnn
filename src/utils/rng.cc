//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/rng.h>

namespace xsdnn {

mm_scalar default_random_engine::rand() {
    __r = this->next_rand(__r);
    return (mm_scalar) __r;
}

mm_scalar default_random_engine::rand(std::random_device &__rd) {
    __r = __rd();
    return this->rand();
}

void default_random_engine::set_seed(int32_t seed) {
    __r = seed ? (seed & __m) : 1;
}

int32_t default_random_engine::next_rand(int32_t r_) {
    uint32_t lo, hi;
    lo = __a * (int32_t) (r_ & 0xFFFF);
    hi = __a * (int32_t) ((uint32_t) r_ >> 16);
    lo += (hi & 0x7FFF) << 16;

    if (lo > __m) {
        lo &= __m;
        ++lo;
    }

    lo += hi >> 15;

    if (lo > __m) {
        lo &= __m;
        ++lo;
    }

    return (int32_t) lo;
}
} // xsdnn