//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/network.h>
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
void network<Net>::save(const std::string filename) {
    net_.save_model(filename, network_name_);
}

template<typename Net>
void network<Net>::load(const std::string filename) {
    net_.load_model(filename);
}

void construct_graph(network<graph>& net,
                     const std::vector<layer*>& input,
                     const std::vector<layer*>& out) {
    net.net_.construct(input, out);
}

} // xsdnn

template class xsdnn::network<xsdnn::sequential>;
template class xsdnn::network<xsdnn::graph>;

