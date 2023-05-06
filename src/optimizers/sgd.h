//
// Created by rozhin on 06.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_SGD_H
#define XSDNN_SGD_H

#include "optimizer_base.h"

namespace xsdnn {

class sgd : public optimizer {
public:
    sgd();
    sgd(mm_scalar alpha, mm_scalar weight_decay);
    sgd(const sgd&);
    sgd& operator=(const sgd&);
    virtual ~sgd();

public:
    void update(const mat_t& dw, mat_t& w) override;

public:
    mm_scalar alpha_;
    mm_scalar weight_decay_;
};

}

#endif //XSDNN_SGD_H
