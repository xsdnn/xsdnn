//
// Created by Andrei R. on 31.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_RNG_HPP
#define XSDNN_RNG_HPP

namespace xsdnn {

class base_random_engine {
public:
    virtual Scalar rand() = 0;
    /*
     * This device need to no determenistic case, when seed changing on each call.
     */
    virtual Scalar rand(std::random_device& __rd) = 0;

    virtual ~base_random_engine() {}
};

class default_random_engine : public base_random_engine{
public:
    explicit default_random_engine(uint32_t seed)
            : __a(16807), __m(2147483647L), __r(seed ? (seed & __m) : 1) {}

    explicit default_random_engine()
            : __a(16807), __m(2147483647L), __r(42) {}

    ~default_random_engine() = default;

    void set_seed(int32_t seed) {
        __r = seed ? (seed & __m) : 1;
    }

    virtual Scalar rand() override{
        __r = this->next_rand(__r);
        return (Scalar) __r;
    }

    virtual Scalar rand(std::random_device& __rd) override {
        __r = __rd();
        return this->rand();
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
