//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "network.h"

namespace xsdnn {

template<typename L>
network& network::operator<<(L &&layer) {
    net_.add(std::forward<L>(layer));
    return *this;
}

void network::init_weight() {
    net_.setup(true);
}

mat_t network::predict(const mat_t &in) {
    return fprop(in);
}

tensor_t network::predict(const tensor_t &in) {
    return fprop(in);
}

std::vector<tensor_t> network::predict(const std::vector<tensor_t> &in) {
    return fprop(in);
}


mat_t network::fprop(const mat_t &in) {
    return fprop(tensor_t{in})[0];
}

tensor_t network::fprop(const tensor_t &in) {
    return fprop(std::vector<tensor_t>{ in })[0];
}

std::vector<tensor_t> network::fprop(const std::vector<tensor_t> &in) {
    return net_.forward(in);
}

} // xsdnn