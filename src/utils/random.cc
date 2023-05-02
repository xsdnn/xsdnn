//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "random.h"

namespace xsdnn {

mm_scalar uniform_distribution::operator()(base_random_engine &eng) {
#if defined(XS_NO_DTRMNST)
    return min_ + eng.rand(rd) * coef_ / RAND_MAX;
#else
    return min_ + eng.rand() * coef_ / RAND_MAX;
#endif
}

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