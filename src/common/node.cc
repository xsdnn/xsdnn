//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/node.h>

namespace xsdnn {

    node::~node() {}

    std::vector<edgeptr_t> &node::prev() {
        return prev_;
    }

    const std::vector<edgeptr_t> &node::prev() const {
        return prev_;
    }

    std::vector<edgeptr_t> &node::next() {
        return next_;
    }

    const std::vector<edgeptr_t> &node::next() const {
        return next_;
    }

    std::vector<node *> node::prev_nodes() const {
        std::vector<node *> vecs;
        for (auto &e : prev_) {
            if (e && e->prev()) {
                vecs.insert(vecs.end(), e->prev());
            }
        }
        return vecs;
    }

    std::vector<node *> node::next_nodes() const {
        std::vector<node *> vecs;
        for (auto &e : next_) {
            if (e) {
                auto n = e->next();
                vecs.insert(vecs.end(), n.begin(), n.end());
            }
        }
        return vecs;
    }

    tensor_t* edge::get_data() {
        return &data_;
    }

    const tensor_t* edge::get_data() const {
        return &data_;
    }

    tensor_t* edge::get_gradient() {
        return &grad_;
    }

    const tensor_t* edge::get_gradient() const {
        return &grad_;
    }

    tensor_type edge::ttype() const {
        return ttype_;
    }

    const shape3d &edge::shape() const {
        return shape_;
    }

    node *edge::prev() {
        return prev_;
    }

    const node *edge::prev() const {
        return prev_;
    }

    std::vector<node *> edge::next() {
        return next_;
    }

    const std::vector<node *> edge::next() const {
        return next_;
    }

    void edge::clear_grads() {
        for (size_t i = 0; i < grad_.size(); ++i) {
            tensorize::fill(grad_[i].data(), grad_[i].size(), 0.0f);
        }
    }

    void edge::add_next_node(node *nd) {
        next_.push_back(nd);
    }

    void edge::accumulate_grads(mat_t* dst) {
        assert(!grad_.empty());
        size_t sample_count = grad_.size();
        size_t size = grad_[0].size();

        if (dst->empty()) (*dst).resize(size);
        tensorize::fill(dst->data(), size, 0.0f);

        for (size_t sample = 0; sample < sample_count; ++sample) {
            const auto& grad_sample = grad_[sample];
            for (size_t i = 0; i < size; ++i) {
                (*dst)[i] += grad_sample[i];
            }
        }
    }
}