//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NODE_H
#define XSDNN_NODE_H

#include <memory>
#include <vector>
#include "../utils/tensor_shape.h"
#include "../utils/util.h"
#include "../utils/tensor.h"
#include "../utils/tensor_utils.h"

namespace xsdnn {

class edge;
using edgeptr_t = std::shared_ptr<edge>;
class layer;

/*
* Containing edge container before layer operation and after
*/
class node {
public:
    node (size_t in_concept, size_t out_concept) : prev_(in_concept), next_(out_concept) {}
    virtual ~node();

    std::vector<edgeptr_t>& prev();

    const std::vector<edgeptr_t>& prev() const;

    std::vector<edgeptr_t>& next();

    const std::vector<edgeptr_t>& next() const;

protected:
    node() = delete;

    friend void connect(layer* last_node, layer* next_node,
                        size_t last_node_data_concept_idx, size_t next_node_data_concept_idx);

    mutable std::vector<edgeptr_t> prev_; // can be weight & bias & data
    mutable std::vector<edgeptr_t> next_; // output
};

/*
* Containing input/output information like (weights & bias & data) & grad
*/
class edge {
public:
    edge(node* prev, const shape3d& shape, tensor_type ttype)
        :   shape_(shape), ttype_(ttype),
            data_({ mat_t(shape.size()) }),
            grad_({ mat_t(shape.size()) }),
            prev_(prev) {}

    tensor_t* get_data();
    const tensor_t* get_data() const;

    tensor_t* get_gradient();
    const tensor_t* get_gradient() const;

    tensor_type ttype() const;
    const shape3d& shape() const;

    node* prev();
    const node* prev() const;

    std::vector<node*> next();
    const std::vector<node*> next() const;

    void clear_grads();
    void add_next_node(node* nd);
    void accumulate_grads(mat_t* dst);

private:
    shape3d shape_;
    tensor_type ttype_;
    tensor_t data_;
    tensor_t grad_;
    node* prev_;                // 'producer'
    std::vector<node*> next_;   // 'consumer'
};

/*
 * Контейнер служит для упаковки нод одного типа,
 * для последующего соединения в общий граф.
 */
template<typename T>
struct node_container {
public:
    node_container(T l1, T l2) {
        nodes_.push_back(l1);
        nodes_.push_back(l2);
    }

    void add_node(T l) {
        nodes_.push_back(l);
    }

    size_t size() const {
        return nodes_.size();
    }

public:
    std::vector<T> nodes_;
};

template<typename T>
node_container<T*> operator , (T& l1, T& l2) {
    return node_container<T*>(&l1, &l2);
}

template<typename T>
node_container<T*> operator , (node_container<T*>& lhs, T& rhs) {
    lhs.add_node(&rhs);
    return lhs;
}

template<typename T, typename U>
U& operator << (const node_container<T*>& lhs, U& rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        connect(lhs.nodes_[i], &rhs, 0, i);
    }
    return rhs;
}

} // xsdnn

#endif //XSDNN_NODE_H
