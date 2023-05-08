//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MACRO_H
#define XSDNN_MACRO_H
#include <iostream>

#define XS_UNUSED_PARAMETER(x) (void)(x)
#define XS_2D_1D_CONVERTER(i, j, lda) i * lda + j

#define XS_REGISTER_LAYER_FOR_NET(layer_type)                                                       \
template<>                                                                                          \
network& network::operator<<(layer_type &&layer) {                                                  \
    net_.owner_nodes_.push_back(std::make_shared<                                                   \
            typename std::remove_reference<layer_type>::type>(layer));                              \
    net_.nodes_.push_back(net_.owner_nodes_.back().get());                                          \
                                                                                                    \
    if (net_.nodes_.size() > 1) {                                                                   \
        auto last_node = net_.nodes_[net_.nodes_.size() - 2];                                       \
        auto next_node = net_.nodes_[net_.nodes_.size() - 1];                                       \
        auto data_idx = find_data_idx(last_node->out_types(), next_node->in_types());               \
        connect(last_node, next_node, data_idx.first, data_idx.second);                             \
    }                                                                                               \
    net_.check_connectivity();                                                                      \
    return *this;                                                                                   \
}

#define XS_REGISTER_WEIGHT_INIT(wi_type)                                                            \
template<>                                                                                          \
void layer::weight_init(const wi_type& f) {                                                         \
    weight_init_ = std::make_shared<wi_type>(f);                                                    \
}                                                                                                   \

#define XS_REGISTER_BIAS_INIT(bi_type)                                                              \
template<>                                                                                          \
void layer::bias_init(const bi_type& f) {                                                           \
    bias_init_ = std::make_shared<bi_type>(f);                                                      \
}                                                                                                   \

#endif //XSDNN_MACRO_H
