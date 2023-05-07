//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "loss_base.h"
#include <cassert>

namespace xsdnn {

    loss::loss() = default;
    loss::loss(const xsdnn::loss &) = default;
    loss &loss::operator=(const xsdnn::loss &) = default;
    loss::~loss() {}

void gradient(loss* l_ptr, const mat_t& y, const mat_t& a, mat_t& dst) {
    l_ptr->df(y, a, dst);
}

void gradient(loss* l_ptr, const tensor_t& y, const tensor_t& a, tensor_t& dst) {
    assert(y.size() == dst.size());
    size_t feature_count = y[0].size();
    for (size_t channel = 0; channel < y.size(); ++channel) {
        dst[channel].resize(feature_count);
        gradient(l_ptr, y[channel], a[channel], dst[channel]);
    }
}

void gradient(loss* l_ptr,
              const std::vector<tensor_t>& y,
              const std::vector<tensor_t>& a,
              std::vector<tensor_t>& dst) {
    size_t sample_count = y.size();
    size_t channel_count = y[0].size();

    assert(y.size() == dst.size());

    for (size_t i = 0; i < sample_count; ++i) {
        assert(y[i].size() == channel_count);
        assert(a[i].size() == channel_count);

        dst[i].resize(channel_count);

        gradient(l_ptr, y[i], a[i], dst[i]);
    }
}

}