//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NODES_H
#define XSDNN_NODES_H

#include "node.h"
#include "optimizers/optimizer_base.h"

namespace xsdnn {

/*
 * Basic class for neural network seq
 */

class nodes {
public:
    typedef std::vector<layer*>::iterator iterator;
    typedef std::vector<layer*>::const_iterator const_iterator;

    virtual
    void
    backward(const std::vector<tensor_t>& start) = 0;

    virtual
    std::vector<tensor_t>
    forward(const std::vector<tensor_t>& start) = 0;

    virtual
    void
    update_weights(optimizer* opt, int batch_size);

    virtual
    void
    setup(bool reset_weight);

    void clear_grads();

    // TODO: дописать все интерфейсные методы
protected:
    std::vector<layer*> nodes_;
};

} // xsdnn

#endif //XSDNN_NODES_H
