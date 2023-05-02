//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NODES_H
#define XSDNN_NODES_H

#include "node.h"

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
    backward(const std::vector<...>& ) = 0;

};

} // xsdnn

#endif //XSDNN_NODES_H
