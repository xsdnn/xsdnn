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

    std::vector<node*> prev_nodes() const;
    std::vector<node*> next_nodes() const;

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

namespace detail {

template<typename T>
class graph_builder {
public:
    graph_builder(T* main_node)
    : main_node_(main_node),
    curr_connected_idx(0) {}

public:
    template<typename ConnectedNode, typename... Args>
    void connect_subgraph(ConnectedNode* node, Args*... args) {
        connect_subgraph(node);
        connect_subgraph(args...);
    }

    template<typename ConnectedNode>
    void connect_subgraph(ConnectedNode* node) {
        connect(node, main_node_, 0, curr_connected_idx);
        curr_connected_idx += 1;
    }

private:
    T* main_node_;
    size_t curr_connected_idx;
};

} // detail

template<typename MainNode, typename... Args>
void connect_subgraph(MainNode& node, Args&... args) {
    detail::graph_builder<MainNode> graphBuilder(&node);
    graphBuilder.template connect_subgraph(&args...);
}

} // xsdnn

#endif //XSDNN_NODE_H
