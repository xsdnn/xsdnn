//
// Created by rozhin on 06.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "sgd.h"

namespace xsdnn {

sgd::sgd() : alpha_(0.0f), weight_decay_(0.0f) {}
sgd::sgd(mm_scalar alpha, mm_scalar weight_decay) : alpha_(alpha), weight_decay_(weight_decay) {}
sgd::sgd(const sgd &) = default;
sgd &sgd::operator=(const sgd &) = default;
sgd::~sgd() = default;

// TODO: можно ли оптимизировать? параллельный расчет, simd и прочее?
void sgd::update(const mat_t &dw, mat_t &w) {
    if (weight_decay_ == 0.0f) {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = w[i] - alpha_ * dw[i];
        }
    } else {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = w[i] - alpha_ * (dw[i] + weight_decay_ * w[i]);
        }
    }
}

} // xsnn
