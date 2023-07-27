//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/network.h>
#include <serializer/cerial.h>
#include <fstream>

namespace xsdnn {

template<typename Net>
void network<Net>::init_weight() {
    net_.setup(true);
}

template<typename Net>
void network<Net>::set_num_threads(size_t num_threads) noexcept {
    net_.user_num_threads_ = num_threads;
}

template<typename Net>
bool network<Net>::empty() const {
    return net_.nodes_.empty();
}

template<typename Net>
mat_t network<Net>::predict(const mat_t &in) {
    return fprop(in);
}

template<typename Net>
tensor_t network<Net>::predict(const tensor_t &in) {
    return fprop(in);
}

template<typename Net>
std::vector<tensor_t> network<Net>::predict(const std::vector<tensor_t> &in) {
    return fprop(in);
}


template<typename Net>
mat_t network<Net>::fprop(const mat_t &in) {
    return fprop(tensor_t{in})[0];
}

template<typename Net>
tensor_t network<Net>::fprop(const tensor_t &in) {
    return fprop(std::vector<tensor_t>{ in })[0];
}

template<typename Net>
std::vector<tensor_t> network<Net>::fprop(const std::vector<tensor_t> &in) {
    return net_.forward(in);
}

template<typename Net>
void network<Net>::train(loss *loss, optimizer *opt, const tensor_t &input,
                    const std::vector<size_t> &label, size_t batch_size, size_t epoch) {
    std::vector<tensor_t> input_tensor, output_tensor;
    newaxis(input, input_tensor);
    label2vec(label, output_tensor);

    fit(loss, opt, input_tensor, output_tensor, batch_size, epoch);
}

template<typename Net>
void network<Net>::train(xsdnn::loss *loss, xsdnn::optimizer *opt, const std::vector<tensor_t> &input,
                         const std::vector<tensor_t> &label, size_t batch_size, size_t epoch) {
    std::vector<tensor_t> input_tensor(input.begin(), input.end());
    std::vector<tensor_t> output_tensor(label.begin(), label.end());
    fit(loss, opt, input_tensor, output_tensor, batch_size, epoch);
}

template<typename Net>
void network<Net>::fit(loss *l_ptr, optimizer *opt_ptr, std::vector<tensor_t> &input,
                  std::vector<tensor_t> &label, size_t batch_size, size_t epoch) {
    net_.setup(true);
    size_t num_threads = net_.user_num_threads_ > 0
                                                ? net_.user_num_threads_
                                                : std::thread::hardware_concurrency() / 2;
    for (auto l : net_) {
        l->set_parallelize(true);
        l->set_num_threads(num_threads);
    }
    opt_ptr->reset();
    for (size_t e = 0; e < epoch; ++e) {
        for (size_t b = 0; b < input.size(); b += batch_size) {
            fit_batch(l_ptr,
                      opt_ptr,
                      &input[b], &label[b],
                      std::min(batch_size, input.size() - b));
        }
    }
}

template<typename Net>
void network<Net>::fit_batch(loss *l_ptr, optimizer *opt_ptr, const tensor_t *input,
                        const tensor_t *label, size_t batch_size) {
    if (batch_size == 1) {
        throw xs_error("Not implemented yet"); // FIXME: доделать
    } else if (batch_size > 1) {
        compute(l_ptr, opt_ptr, input, label, batch_size);
    } else {
        throw xs_error("Batch size <= 0");
    }
}

template<typename Net>
void network<Net>::compute(loss *l_ptr, optimizer *opt_ptr, const tensor_t *input,
                      const tensor_t *label, size_t batch_size) {
    std::vector<tensor_t> in_batch(&input[0], &input[0] + batch_size);
    std::vector<tensor_t> l_batch(&label[0], &label[0] + batch_size);
    bprop(l_ptr, opt_ptr, fprop(in_batch), l_batch);
}

template<typename Net>
void network<Net>::bprop(loss* l_ptr,
                    optimizer* opt_ptr,
                    const std::vector<tensor_t> &net_out,
                    const std::vector<tensor_t> &label) {
    std::vector<tensor_t> delta; // grad(loss)
    delta.resize(net_out.size());
    gradient(l_ptr, net_out, label, delta);
    net_.backward(delta);
    net_.update_weights(opt_ptr);
}

template<typename Net>
void network<Net>::newaxis(const tensor_t &in, std::vector<tensor_t> &out) {
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out.emplace_back(tensor_t{ in[i] });
    }
}

template<typename Net>
void network<Net>::label2vec(const std::vector<size_t> &label,
                        std::vector<tensor_t>& output) {
    size_t num_label = label.size();
    size_t outdim = net_.out_data_size();

    tensor_t predef_vec;
    predef_vec.reserve(num_label);
    for (size_t i = 0; i < num_label; ++i) {
        assert(label[i] < outdim);
        mm_scalar min_output_val = net_.nodes_.back()->out_value_range().first;
        mm_scalar max_output_val = net_.nodes_.back()->out_value_range().second;
        predef_vec.emplace_back(outdim, min_output_val);
        predef_vec[i][label[i]] = max_output_val;
    }

    newaxis(predef_vec, output);
}

template<typename Net>
void network<Net>::save(const std::string filename) const {
    layer_register();

    xs::GraphInfo* graph = new xs::GraphInfo;

    xs::NodeInfo* node;
    xs::TensorInfo* tensor;

    for (size_t i = 0; i < net_.nodes_.size(); ++i) {
        node = graph->add_nodes();
        tensor = graph->add_tensors();
        serializer::get_instance().save(node, tensor, net_.nodes_[i]);
    }
    xs::ModelInfo model;
    model.template set_name(network_name_);
    model.set_allocated_graph(graph);

    std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
    model.SerializeToOstream(&ofs);
}

template<typename Net>
void network<Net>::load(const std::string filename) {
    std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);
    if (!ifs.is_open()) {
        xs_error("Error when opening model_filename file.");
    }
    xs::ModelInfo model;
    if (!model.ParseFromIstream(&ifs)) {
        xs_error("Error when parse model.");
    }

    xs::GraphInfo model_graph = model.graph();
    // TODO: как по названию ноды добавлять в сетку?

    net_.nodes_.clear();
    net_.owner_nodes_.clear();

    for (size_t i = 0; i < model_graph.nodes_size(); ++i) {
        serializer::get_instance().load(&model_graph.nodes(i),
                                        &model_graph.tensors(i),
                                        net_.owner_nodes_);
    }
}

void construct_graph(network<graph>& net,
                     const std::vector<layer*>& input,
                     const std::vector<layer*>& out) {
    net.net_.construct(input, out);
}

} // xsdnn

template class xsdnn::network<xsdnn::sequential>;
template class xsdnn::network<xsdnn::graph>;

