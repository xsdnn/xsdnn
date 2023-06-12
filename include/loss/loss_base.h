//
// Created by rozhin on 05.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_LOSS_BASE_H
#define XSDNN_LOSS_BASE_H

#include <utils/tensor.h>

namespace xsdnn {

class loss {
public:
    loss();
    loss(const loss&);
    loss& operator=(const loss&);
    virtual ~loss();

public:
    virtual mm_scalar f(const mat_t& y, const mat_t& a) = 0;
    virtual void df(const mat_t& y, const mat_t& a, mat_t& dst) = 0;
};

void gradient(loss* l_ptr, const mat_t& y, const mat_t& a, mat_t& dst);
void gradient(loss* l_ptr, const tensor_t& y, const tensor_t& a, tensor_t& dst);
void gradient(loss* l_ptr,
              const std::vector<tensor_t>& y,
              const std::vector<tensor_t>& a,
              std::vector<tensor_t>& dst);


} // xsdnn

#endif //XSDNN_LOSS_BASE_H
