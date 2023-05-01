//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <mmpack/mmpack.h>
#include <random>
#include "rng.h"

namespace xsdnn {

class base_distribution {
public:
    virtual ~base_distribution() {}

protected:
    std::random_device rd;
};

class uniform_distribution : protected base_distribution {
public:
    uniform_distribution(mm_scalar min, mm_scalar max) : min_(min), max_(max), coef_(max_ - min_) {}

#if defined(XS_NO_DTRMNST)
    mm_scalar operator() (base_random_engine& eng) {
    return min_ + eng.rand(rd) * coef_ / RAND_MAX;
}
#else
    mm_scalar operator() (base_random_engine& eng) {
        return min_ + eng.rand() * coef_ / RAND_MAX;
    }
#endif

private:
    mm_scalar min_;
    mm_scalar max_;
    mm_scalar coef_;
};

mm_scalar uniform_rand(mm_scalar min, mm_scalar max) {
    uniform_distribution dst(min, max);
    default_random_engine eng;
    return dst(eng);
}

void uniform_rand(mm_scalar* data, size_t size, mm_scalar min, mm_scalar max) {
    uniform_distribution dst(min, max);
    default_random_engine eng;
    for (size_t i = 0; i < size; ++i) {
        data[i] = dst(eng);
    }
}

} // xsdnn