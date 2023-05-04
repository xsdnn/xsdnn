//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NODES_H
#define XSDNN_NODES_H

#include "node.h"
#include "../layers/layer.h"
#include "../optimizers/optimizer_base.h"

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

    size_t size() const;
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    layer* operator[] (size_t index);
    const layer* operator[] (size_t index) const;
    size_t in_data_size() const;
    size_t out_data_size() const;

protected:
    // TODO: Impl push back method
    template<typename T>
    void push_back(T&& node);

    // TODO: реализовать без копирования - возможно ли это?
    std::vector<tensor_t> reorder_input(const std::vector<tensor_t>& input);

protected:
    std::vector<layer*> nodes_;
};

class sequential : public nodes {
public:
    virtual void backward(const std::vector<tensor_t>& start);
    virtual std::vector<tensor_t> forward(const std::vector<tensor_t>& start);

    template<typename T>
    void add(T&& layer);

    void check_connectivity();

protected:
    std::vector<tensor_t> reorder_output(const std::vector<tensor_t>& output);
};

} // xsdnn

#endif //XSDNN_NODES_H
