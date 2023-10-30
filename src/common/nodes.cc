//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/nodes.h>
#include <utils/graph.h>
#include <utils/macro.h>
#include <unordered_map>
#include <algorithm>
#include <serializer/cerial.h>
#include <fstream>
#include <thread>

namespace xsdnn {

    nodes::nodes() {}
    nodes::~nodes()  {}

    void nodes::setup(bool reset_weight) {
        for (auto l : nodes_) {
            l->setup(reset_weight);
        }
    }

    void nodes::set_num_threads() {
        for (auto* l : nodes_) {
            l->set_num_threads(user_num_threads_ != 0 ? user_num_threads_ : std::thread::hardware_concurrency() / 2);
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

        xs::GraphInfo* Graph = new xs::GraphInfo;

        xs::NodeInfo* node;
        xs::TensorInfo* tensor;

        for (size_t i = 0; i < nodes_.size(); ++i) {
            node = Graph->add_nodes();
            tensor = Graph->add_tensors();
            serializer::get_instance().save(node, tensor, nodes_[i]);
        }

        if (typeid(*this) == typeid(sequential)) {
            dynamic_cast<sequential *>(this)->save_connections(Graph);
        } else {
            dynamic_cast<graph *>(this)->save_connections(Graph);
        }

        xs::ModelInfo model;
        model.set_name(network_name_);
        model.set_allocated_graph(Graph);

        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
        model.SerializeToOstream(&ofs);
    }

    void nodes::load_model(const std::string& filename) {
        layer_register();
        std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);
        if (!ifs.is_open()) {
            std::string msg = "Error when opening \x1B[33m" + filename;
            throw xs_error(msg);
        }
        xs::ModelInfo model;
        if (!model.ParseFromIstream(&ifs)) {
            throw xs_error("Error when parse model.");
        }

        xs::GraphInfo model_graph = model.graph();

        nodes_.clear();
        owner_nodes_.clear();

        for (size_t i = 0; i < static_cast<size_t>(model_graph.nodes_size()); ++i) {
            serializer::get_instance().load(&model_graph.nodes(i),
                                            &model_graph.tensors(i),
                                            owner_nodes_);
        }

        for (auto &n : owner_nodes_) {
            nodes_.push_back(&*n);
        }
        if (typeid(*this) == typeid(sequential)) {
            dynamic_cast<sequential *>(this)->load_connections(&model_graph);
        } else {
            dynamic_cast<graph *>(this)->load_connections(&model_graph);
        }
    }

    bool nodes::have_engine_xnnpack() {
        for (const auto* l : nodes_) {
            if (l->engine() == core::backend_t::xnnpack) return true;
        } return false;
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
                connection_mismatch(prev[data_idx.first], next[data_idx.second]);
            }
        }
    }

    void sequential::connection_mismatch(xsdnn::shape3d lhs, xsdnn::shape3d rhs) {
        std::ostringstream io;
        io << "\x1B[31m" << "Critical Error! Shape mismatch!" << std::endl;

        io << "\x1B[33m" << "out shape:\t" << lhs << std::endl;

        io << "\x1B[33m" << "in  shape:\t" << rhs << std::endl;

        std::string message = io.str();
        throw xs_error(message.c_str());
    }

    void sequential::reorder_output(const std::vector<tensor_t> &input, std::vector<tensor_t> &output) {
        const size_t sample_count = input[0].size();
        output.resize(sample_count, tensor_t(1));

        for (size_t sample = 0; sample < sample_count; ++sample) {
            output[sample][0] = (input[0])[sample];
        }
    }

    void sequential::load_connections(xs::GraphInfo* Graph) {
        for(size_t i = 0; i < nodes_.size() - 1; ++i) {
            auto last_node = nodes_[i];
            auto next_node = nodes_[i + 1];
            auto data_idx = find_data_idx(last_node->out_types(), next_node->in_types());
            connect(last_node, next_node, data_idx.first, data_idx.second);
        }
    }

    void sequential::save_connections(xs::GraphInfo* Graph) {}

    graph::graph() {}
    graph::~graph() {}

    std::vector<tensor_t> graph::forward(const std::vector<tensor_t> &start) {
        if (have_engine_xnnpack()) {
#ifdef XS_USE_XNNPACK
            core::XNNCompiler::getInstance().initialize();
#else
      throw xs_error("This xsdnn build doesn't support XNNPACK backend engine. Recompile with xsdnn_BUILD_XNNPACK_ENGINE=ON");
#endif
        }
        this->set_num_threads();

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

    void graph::save_connections(xs::GraphInfo *Graph) {
        std::unordered_map<node*, size_t> nd_idx;
        size_t idx = 0;

        for (auto nd : nodes_) {
            nd_idx[nd] = idx++;
        }

        for (auto nd: input_layers_) {
            Graph->add_inputs(nd_idx[nd]);
        }

        for (auto nd: output_layers_) {
            Graph->add_outputs(nd_idx[nd]);
        }

        for (auto l : input_layers_) {
            dfs_graph_traverse(l,
                               [](layer& l) { XS_UNUSED_PARAMETER(l); },
                               [&](edge& e) {
               auto next         = e.next();
               size_t head_index = e.prev()->next_port(e);

               for (auto n : next) {
                   size_t tail_index = n->prev_port(e);
                   xs::ConnectionInfo* connect = Graph->add_connections();
                   connect->set_last_node_idx(nd_idx[e.prev()]);
                   connect->set_next_node_idx(nd_idx[n]);
                   connect->set_last_node_data_concept_idx(head_index);
                   connect->set_next_node_data_concept_idx(tail_index);
               }
            });
        }
    }

    void graph::load_connections(xs::GraphInfo *Graph) {
        for (size_t i = 0; i < Graph->connections_size(); ++i) {
            const xs::ConnectionInfo& c_ = Graph->connections(i);
            connect(nodes_[c_.last_node_idx()],
                    nodes_[c_.next_node_idx()],
                    c_.last_node_data_concept_idx(),
                    c_.next_node_data_concept_idx());
        }

        for (size_t i = 0; i < Graph->inputs_size(); ++i) {
            input_layers_.push_back(nodes_[Graph->inputs(i)]);
        }

        for (size_t i = 0; i < Graph->outputs_size(); ++i) {
            output_layers_.push_back(nodes_[Graph->outputs(i)]);
        }
    }

} // xsdnn