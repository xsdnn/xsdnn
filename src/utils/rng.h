//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_RNG_H
#define XSDNN_RNG_H

#include <mmpack/mmpack.h>
#include <random>
using namespace mmpack;

namespace xsdnn {

class base_random_engine {
public:
    virtual mm_scalar rand() = 0;
    /*
     * This device need to no determenistic case, when seed changing on each call.
     */
    virtual mm_scalar rand(std::random_device& __rd) = 0;

    virtual ~base_random_engine() {}
};

class default_random_engine : public base_random_engine {
public:
    explicit default_random_engine(uint32_t seed)
            : __a(16807), __m(2147483647L), __r(seed ? (seed & __m) : 1) {}

    explicit default_random_engine()
            : __a(16807), __m(2147483647L), __r(42) {}

    ~default_random_engine() = default;

    void set_seed(int32_t seed);
    virtual mm_scalar rand() override;
    virtual mm_scalar rand(std::random_device& __rd) override;

private:
    const uint32_t __a;
    const uint32_t __m;
    int32_t __r;

    int32_t next_rand(int32_t r_);
};

} // xsdnn

#endif //XSDNN_RNG_H
