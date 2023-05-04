//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NETWORK_H
#define XSDNN_NETWORK_H

#include <vector>
#include "../layers/layer.h"
#include "nodes.h"

namespace xsdnn {

class network {
public:
    typedef std::vector<layer*>::iterator iterator;
    typedef std::vector<layer*>::const_iterator const_iterator;

    network() = default;
    network(const network&) = default;
    network& operator=(const network&) = default;
    ~network() = default;

    template<typename L>
    network& operator<<(L &&layer);

public:
    void init_weight();
    mat_t predict(const mat_t& in);
    tensor_t predict(const tensor_t& in);
    std::vector<tensor_t> predict(const std::vector<tensor_t>& in);

protected:
    mat_t fprop(const mat_t& in);
    std::vector<mat_t> fprop(const std::vector<mat_t>& in);
    std::vector<tensor_t> fprop(const std::vector<tensor_t>& in);

private:
    sequential net_;
};

} // xsdnn

#endif //XSDNN_NETWORK_H
