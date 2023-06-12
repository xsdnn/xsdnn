//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_OPTIMIZER_BASE_H
#define XSDNN_OPTIMIZER_BASE_H

#include "../utils/tensor.h"

namespace xsdnn {

class optimizer {
public:
    optimizer();
    optimizer(const optimizer& rhs);
    optimizer& operator=(const optimizer& rhs);

    virtual ~optimizer();
    virtual void update(const mat_t& dw, mat_t& w) = 0;
    virtual void reset();
};

}

#endif //XSDNN_OPTIMIZER_BASE_H
