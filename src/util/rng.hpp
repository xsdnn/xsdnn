//
// Created by Andrei R. on 31.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_RNG_HPP
#define XSDNN_RNG_HPP

namespace xsdnn {

class default_random_engine {
public:
    explicit default_random_engine(uint32_t seed)
            : __a(16807), __m(2147483647L), __r(seed ? (seed & __m) : 1) {}

    ~default_random_engine() = default;

    void set_seed(int32_t seed) {
        __r = seed ? (seed & __m) : 1;
    }

    Scalar rand() {
        __r = this->next_rand(__r);
        return (Scalar) __r;
    }

private:
    const uint32_t __a;
    const uint32_t __m;
    int32_t __r;

    int32_t next_rand(int32_t r_) {
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
};

} // xsdnn

#endif //XSDNN_RNG_HPP
