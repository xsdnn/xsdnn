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
    assert((y.size() == a.size()) == dst.size());

    for (size_t sample = 0; sample < y.size(); ++sample) {
        gradient(l_ptr, y[sample], a[sample], dst[sample]);
    }
}

}