//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NETWORK_H
#define XSDNN_NETWORK_H

#include <vector>
#include <thread>
#include "../layers/layer.h"
#include "nodes.h"
#include "config.h"

namespace xsdnn {

class network {
public:
    typedef std::vector<layer*>::iterator iterator;
    typedef std::vector<layer*>::const_iterator const_iterator;

    network() : network_name_("default_name") {}
    explicit network(const std::string network_name) : network_name_(network_name) {}
    network(const network&) = default;
    network& operator=(const network&) = default;
    ~network() = default;

    const layer *operator[](size_t index) const { return net_[index]; }
    layer *operator[](size_t index) { return net_[index]; }

public:
    void init_weight();
    void set_num_threads(size_t num_threads) noexcept;
    bool empty() const;

    void configure();

    mat_t predict(const mat_t& in);
    tensor_t predict(const tensor_t& in);

    void save(const std::string filename);
    void load(const std::string filename);

protected:
    friend bool operator == (network& lhs, network& rhs) {
        /*
         * Check topological sorted vector
         */
        if (lhs.net_.size() != rhs.net_.size()) {
            return false;
        }

        for (size_t i = 0; i <  lhs.net_.size(); ++i) {
            layer* lhs_layer = lhs.net_[i];
            layer* rhs_layer = rhs.net_[i];
            if (lhs_layer->layer_type() != rhs_layer->layer_type()) {
                return false;
            }
        }

        /*
         * Check weights data
         */

        for (size_t i = 0; i <  lhs.net_.size(); ++i) {
            layer* lhs_layer = lhs.net_[i];
            layer* rhs_layer = rhs.net_[i];

            std::vector<mat_t*> lhs_weights = lhs_layer->weights();
            std::vector<mat_t*> rhs_weights = rhs_layer->weights();

            if (lhs_weights.size() != rhs_weights.size()) {
                return false;
            }

            for (size_t j = 0; j < lhs_weights.size(); ++j) {
                if (*lhs_weights[j] != *rhs_weights[j]) {
                    return false;
                }
            }
        }
        return true;
    }

    friend void construct_graph(network& net,
                                const std::vector<layer*>& input,
                                const std::vector<layer*>& out);

protected:
    void fprop(const mat_t& in);
    void fprop(const std::vector<mat_t>& in);

public:
    graph net_;
    std::string network_name_;
    bool configured_{false};
    friend class InfSession;
};

    void construct_graph(network& net,
                         const std::vector<layer*>& input,
                         const std::vector<layer*>& out);

} // xsdnn

#endif //XSDNN_NETWORK_H
