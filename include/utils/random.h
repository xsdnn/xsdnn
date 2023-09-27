#pragma once
//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../core/framework/tensor.h"
#include "xs_error.h"
#include "rng.h"

namespace xsdnn {

class base_distribution {
public:
    virtual ~base_distribution() {}

protected:
    std::random_device rd;
};

template<typename T>
class uniform_distribution : protected base_distribution {
public:
    uniform_distribution(T min, T max) : min_(min), max_(max), coef_(max_ - min_) {}
    T operator() (base_random_engine& eng) {
#if defined(XS_NO_DTRMNST)
        return min_ + eng.rand(rd) * coef_ / RAND_MAX;
#else
        return min_ + eng.rand() * coef_ / RAND_MAX;
#endif
    }

private:
    T min_;
    T max_;
    T coef_;
};

template<typename T>
T uniform_rand(T min, T max) {
    uniform_distribution dst(min, max);
    default_random_engine eng;
    return dst(eng);
}

template<typename T>
void uniform_rand(tensor_t* data, T min, T max) {
    uniform_distribution dst(min, max);
    default_random_engine eng;
    if (data->dtype() == XsDtype::F32) {
        gsl::span<float> TensorSpan = data->template GetMutableDataAsSpan<float>();
        for (size_t i = 0; i < TensorSpan.size(); ++i) {
            TensorSpan[i] = dst(eng);
        }
    } else {
        throw xs_error("[tensorize fill] Unsupported tensor dtype");
    }
}

} // xsdnn