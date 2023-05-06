//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MSE_LOSS_H
#define XSDNN_MSE_LOSS_H

#include "loss_base.h"

namespace xsdnn {

class mse_loss : public loss {
public:
    mse_loss();
    mse_loss(const mse_loss&);
    mse_loss& operator=(const mse_loss&);
    ~mse_loss() override;

public:
    mm_scalar f(const mat_t& y, const mat_t& a) override;
    void df(const mat_t& y, const mat_t& a, mat_t& dst) override;
};

} // xsdnn

#endif //XSDNN_MSE_LOSS_H
