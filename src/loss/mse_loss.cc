//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "mse_loss.h"
#include <cassert>

namespace xsdnn {

mse_loss::mse_loss() = default;
mse_loss::mse_loss(const mse_loss &) = default;
mse_loss &mse_loss::operator=(const mse_loss &) = default;
mse_loss::~mse_loss() = default;

mm_scalar mse_loss::f(const mat_t &y, const mat_t &a) {
    assert(y.size() == a.size());
    mm_scalar res = 0.0f;

    for (size_t i = 0; i < y.size(); ++i) {
        res += (y[i] - a[i]) * (y[i] - a[i]);
    }
    return res / y.size();
}

void mse_loss::df(const mat_t &y, const mat_t &a, mat_t &dst) {
    assert( (y.size() == a.size()) == dst.size());
    mm_scalar alpha = (mm_scalar) 2.0f / y.size();

    for (size_t i = 0; i < y.size(); ++i) {
        dst[i] = alpha * (y[i] - a[i]);
    }
}

} // xsdnn