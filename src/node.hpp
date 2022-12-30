//
// Created by Andrei R. on 13.10.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_NODE_H
#define XSDNN_NODE_H

namespace xsdnn {

class edge;
using edgeptr_t = std::shared_ptr<edge>;

/*
 * Containing edge container before layer operation and after
 */
class node {
public:
    node (size_t in_concept, size_t out_concept) : prev_(in_concept), next_(out_concept) {}
    node() = delete;
    virtual ~node() {}

    std::vector<edgeptr_t>& prev() {
        return prev_;
    }

    const std::vector<edgeptr_t>& prev() const {
        return prev_;
    }

    std::vector<edgeptr_t>& next() {
        return next_;
    }

    const std::vector<edgeptr_t>& next() const {
        return next_;
    }

protected:
    friend void connect(node* head,
                        node* tail);

    mutable std::vector<edgeptr_t> prev_; // can be weight & bias & data
    mutable std::vector<edgeptr_t> next_; // output
};

/*
 * Containing input/output information like (weights & bias & data) & grad
 */
class edge {
public:
    edge(node* prev, const shape3d& shape, tensor_type ttype) :
        shape_(shape),
        ttype_(ttype),
        data_(shape_.shape()),
        grad_(shape_.shape()),
        prev_(prev)
    {}

    Tensor_3D& get_data() {
        return data_;
    }

    const Tensor_3D& get_data() const {
        return data_;
    };

    Tensor_3D& get_gradient() {
        return grad_;
    }

    const Tensor_3D& get_gradient() const {
        return grad_;
    }

    const tensor_type ttype() const {
        return ttype_;
    }

    const shape3d& shape() const {
        return shape_;
    }

    node* prev() {
        return prev_;
    }

    const node* prev() const {
        return prev_;
    }

    std::vector<node*> next() {
        return next_;
    }

    const std::vector<node*> next() const {
        return next_;
    }

private:
    shape3d shape_;
    tensor_type ttype_;
    Tensor_3D data_;
    Tensor_3D grad_;
    node* prev_;                // 'producer'
    std::vector<node*> next_;   // 'consumer'
};

} // xsdnn

#endif //XSDNN_NODE_H
