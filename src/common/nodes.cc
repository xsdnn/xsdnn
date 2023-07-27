//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/nodes.h>
#include <unordered_map>
#include <algorithm>
#include <serializer/cerial.h>
#include <fstream>

namespace xsdnn {

    nodes::nodes() {}
    nodes::~nodes()  {}

    void nodes::update_weights(optimizer *opt) {
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

    void nodes::save_model(const std::string& filename,
                           const std::string& network_name_) {
        layer_register();

        xs::GraphInfo* graph = new xs::GraphInfo;

        xs::NodeInfo* node;
        xs::TensorInfo* tensor;

        for (size_t i = 0; i < nodes_.size(); ++i) {
            node = graph->add_nodes();
            tensor = graph->add_tensors();
            serializer::get_instance().save(node, tensor, nodes_[i]);
        }
        xs::ModelInfo model;
        model.set_name(network_name_);
        model.set_allocated_graph(graph);

        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
        model.SerializeToOstream(&ofs);

        if (typeid(*this) == typeid(sequential)) {
            dynamic_cast<sequential *>(this)->save_connections();
        } else {
            throw xs_error("NotImplementedYet");
        }
    }

    void nodes::load_model(const std::string& filename) {
        std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);
        if (!ifs.is_open()) {
            xs_error("Error when opening model_filename file.");
        }
        xs::ModelInfo model;
        if (!model.ParseFromIstream(&ifs)) {
            xs_error("Error when parse model.");
        }

        xs::GraphInfo model_graph = model.graph();

        nodes_.clear();
        owner_nodes_.clear();

        for (size_t i = 0; i < model_graph.nodes_size(); ++i) {
            serializer::get_instance().load(&model_graph.nodes(i),
                                            &model_graph.tensors(i),
                                            owner_nodes_);
        }

        for (auto &n : owner_nodes_) {
            nodes_.push_back(&*n);
        }
        if (typeid(*this) == typeid(sequential)) {
            dynamic_cast<sequential *>(this)->load_connections();
        } else {
            throw xs_error("NotImplementedYet");
        }
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

    void sequential::load_connections() {
        for(size_t i = 0; i < nodes_.size() - 1; ++i) {
            auto last_node = nodes_[i];
            auto next_node = nodes_[i + 1];
            auto data_idx = find_data_idx(last_node->out_types(), next_node->in_types());
            connect(last_node, next_node, data_idx.first, data_idx.second);
        }
    }

    void sequential::save_connections() {}

    graph::graph() {}
    graph::~graph() {}

    std::vector<tensor_t> graph::forward(const std::vector<tensor_t> &start) {
        size_t input_data_concept_count = start[0].size();

        if (input_data_concept_count != input_layers_.size()) {
            throw xs_error("input size mismatch");
        }

        std::vector<tensor_t> reordered_data;
        reorder_input(start, reordered_data);
        assert(reordered_data.size() == input_data_concept_count);

        for (size_t channel_index = 0; channel_index < input_data_concept_count; channel_index++) {
            input_layers_[channel_index]->set_in_data({ reordered_data[channel_index] });
        }

        for (auto l : nodes_) {
            l->forward();
        }
        std::vector<tensor_t> out;
        reorder_output(out);
        return out;
    }

    void graph::backward(const std::vector<tensor_t> &start) {
        size_t output_data_concept_count = start[0].size();

        if (output_data_concept_count != output_layers_.size()) {
            throw xs_error("input size mismatch");
        }

        std::vector<tensor_t> reordered_grad;
        reorder_input(start, reordered_grad);
        assert(reordered_grad.size() == output_data_concept_count);

        for (size_t i = 0; i < output_data_concept_count; i++) {
            output_layers_[i]->set_out_grads({ reordered_grad[i] });
        }

        for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
            (*l)->backward();
        }
    }

    void graph::construct(const std::vector<layer *> &input,
                        const std::vector<layer *> &output) {
        std::vector<layer *> sorted;
        std::vector<node *> input_nodes(input.begin(), input.end());
        std::unordered_map<node *, std::vector<uint8_t>> removed_edge;

        while (!input_nodes.empty()) {
            sorted.push_back(dynamic_cast<layer *>(input_nodes.back()));
            input_nodes.pop_back();

            layer *curr              = sorted.back();
            std::vector<node *> next = curr->next_nodes();

            for (size_t i = 0; i < next.size(); i++) {
                if (!next[i]) continue;
                // remove edge between next[i] and current
                if (removed_edge.find(next[i]) == removed_edge.end()) {
                    removed_edge[next[i]] =
                            std::vector<uint8_t>(next[i]->prev_nodes().size(), 0);
                }

                std::vector<uint8_t> &removed = removed_edge[next[i]];
                removed[find_index(next[i]->prev_nodes(), curr)] = 1;

                if (std::all_of(removed.begin(), removed.end(),
                                [](uint8_t x) { return x == 1; })) {
                    input_nodes.push_back(next[i]);
                }
            }
        }

        for (auto& n : sorted) {
            nodes_.push_back(n);
        }

        input_layers_ = input;
        output_layers_ = output;

        setup(false);
    }

    size_t graph::find_index(const std::vector<node *> &nodes, layer *target) {
        for (size_t i = 0; i < nodes.size(); i++) {
            if (nodes[i] == static_cast<node *>(&*target)) return i;
        }
        throw xs_error("invalid connection");
    }

    void graph::reorder_output(std::vector<tensor_t> &output) {
        std::vector<tensor_t> out;
        size_t output_channel_count = output_layers_.size();
        for (size_t output_channel = 0; output_channel < output_channel_count;
             ++output_channel) {
            out = output_layers_[output_channel]->output();

            size_t sample_count = out[0].size();
            if (output_channel == 0) {
                assert(output.empty());
                output.resize(sample_count, tensor_t(output_channel_count));
            }

            assert(output.size() == sample_count);

            for (size_t sample = 0; sample < sample_count; ++sample) {
                output[sample][output_channel] = out[0][sample];
            }
        }
    }

} // xsdnn