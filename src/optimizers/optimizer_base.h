//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_OPTIMIZER_BASE_H
#define XSDNN_OPTIMIZER_BASE_H

namespace xsdnn {

class optimizer {
public:
    optimizer() = default;
    optimizer(const optimizer& rhs) = default;
    optimizer& operator=(const optimizer& rhs) = default;

    virtual ~optimizer() = default;
    virtual void update(const mat_t& dw, mat_t& w) = 0;
    virtual void reset() {}
};

}

#endif //XSDNN_OPTIMIZER_BASE_H
