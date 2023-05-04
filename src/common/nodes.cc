//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "nodes.h"

namespace xsdnn {

    nodes::nodes() {}
    nodes::~nodes()  {}

    void nodes::update_weights(optimizer *opt, int batch_size) {
        for (auto l : nodes_) {
            l->update_weight(opt);
        }
    }

    void nodes::setup(bool reset_weight) {
        for (auto l : nodes_) {
            l->setup(reset_weight);
        }
    }

    void nodes::clear_grads() {
        for (auto l : nodes_) {
            l->clear_grads();
        }
    }

    size_t nodes::size() const {
        return nodes_.size();
    }

    nodes::iterator nodes::begin() {
        return nodes_.begin();
    }

    nodes::iterator nodes::end() {
        return nodes_.end();
    }

    nodes::const_iterator nodes::begin() const {
        return nodes_.begin();
    }

    nodes::const_iterator nodes::end() const {
        return nodes_.begin();
    }

    layer *nodes::operator[](size_t index) {
        return nodes_[index];
    }

    const layer *nodes::operator[](size_t index) const {
        return nodes_[index];
    }

    size_t nodes::in_data_size() const {
        return nodes_.front()->in_data_size();
    }

    size_t nodes::out_data_size() const {
        return nodes_.back()->out_data_size();
    }

    template<typename T>
    void nodes::push_back(T &&node) {
        owner_nodes_.push_back(std::make_shared<
                typename std::remove_reference<T>::type>(node));
        nodes_.push_back(owner_nodes_.back().get());
    }

    void nodes::reorder_input(const std::vector<tensor_t> &input,
                              std::vector<tensor_t> &output) {
        size_t sample_count  = input.size();
        size_t channel_count = input[0].size();

        output.resize(channel_count);
        for (size_t i = 0; i < channel_count; ++i) {
            output[i].resize(sample_count);
        }

        for (size_t sample = 0; sample < sample_count; ++sample) {
            assert(input[sample].size() == channel_count);
            for (size_t channel = 0; channel < channel_count; ++channel) {
                output[channel][sample] = input[sample][channel];
            }
        }
    }

    sequential::sequential() {}
    sequential::~sequential() {}

    void sequential::backward(const std::vector<tensor_t> &start) {
        std::vector<tensor_t> reorder_grad;
        reorder_input(start, reorder_grad);
        nodes_.back()->set_out_grads(reorder_grad); // FIXME: проблема может быть здесь

        for (auto l = nodes_.rbegin(); l != nodes_.rend(); ++l) {
            (*l)->backward();
        }
    }

    std::vector<tensor_t> sequential::forward(const std::vector<tensor_t> &start) {
        std::vector<tensor_t> reorder_data;
        reorder_input(start, reorder_data);
        nodes_.front()->set_in_data(reorder_data);

        for (auto l = nodes_.begin(); l != nodes_.end(); ++l) {
            (*l)->forward();
        }

        std::vector<tensor_t> output;
        reorder_output(nodes_.back()->output(), output);
        return output;
    }

    template<typename T>
    void sequential::add(T &&layer) {
        push_back(layer);

        if (nodes_.size() > 1) {
            auto last_node = nodes_[nodes_.size() - 2];
            auto next_node = nodes_[nodes_.size() - 1];
            auto data_idx = find_data_idx(last_node->out_types(), next_node->in_types());
            connect(last_node, next_node, data_idx.first, data_idx.second);
        }
        check_connectivity();
    }

    void sequential::check_connectivity() {
        for (size_t i = 0; i < nodes_.size() - 1; ++i) {
            auto data_idx = find_data_idx(nodes_[i]->out_types(), nodes_[i]->in_types());
            auto prev = nodes_[i]->out_shape();
            auto next = nodes_[i+1]->in_shape();

            if (prev[data_idx.first] != next[data_idx.second]) {
                throw xs_error("Connected error");
            }
        }
    }

    void sequential::reorder_output(const std::vector<tensor_t> &input, std::vector<tensor_t> &output) {
        const size_t sample_count = input[0].size();
        output.resize(sample_count, tensor_t(1));

        for (size_t sample = 0; sample < sample_count; ++sample) {
            output[sample][0] = (input[0])[sample];
        }
    }

} // xsdnn