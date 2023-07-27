//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NETWORK_H
#define XSDNN_NETWORK_H

#include <vector>
#include <thread>
#include "../layers/layer.h"
#include "../loss/loss_base.h"
#include "../optimizers/optimizer_base.h"
#include "nodes.h"
#include "config.h"

namespace xsdnn {

template<typename Net>
class network {
public:
    typedef std::vector<layer*>::iterator iterator;
    typedef std::vector<layer*>::const_iterator const_iterator;

    explicit network(const std::string network_name) : network_name_(network_name) {}
    network(const network&) = default;
    network& operator=(const network&) = default;
    ~network() = default;

    template<typename L>
    network& operator<<(L &&layer) {
        net_.owner_nodes_.push_back(std::make_shared<typename std::remove_reference<L>::type>(layer));
        net_.nodes_.push_back(net_.owner_nodes_.back().get());

        if (net_.nodes_.size() > 1) {
            auto last_node = net_.nodes_[net_.nodes_.size() - 2];
            auto next_node = net_.nodes_[net_.nodes_.size() - 1];
            auto data_idx = find_data_idx(last_node->out_types(), next_node->in_types());
            connect(last_node, next_node, data_idx.first, data_idx.second);
        }
        net_.check_connectivity();
        return *this;
    }

public:
    void init_weight();
    void set_num_threads(size_t num_threads) noexcept;
    bool empty() const;

    mat_t predict(const mat_t& in);
    tensor_t predict(const tensor_t& in);
    std::vector<tensor_t> predict(const std::vector<tensor_t>& in);

    /*
     * For sequency execution
     */
    void train(loss* loss,
               optimizer* opt,
               const tensor_t& input,
               const std::vector<size_t>& label,
               size_t batch_size,
               size_t epoch);


    /*
     * For graph execution
     */
    void train(loss* loss,
               optimizer* opt,
               const std::vector<tensor_t>& input, /*multiple input's*/
               const std::vector<tensor_t>& label, /*multiple output's*/
               size_t batch_size,
               size_t epoch);

    void save(const std::string filename) const;
    void load(const std::string filename);

protected:
    void fit(loss* l_ptr,
             optimizer* opt_ptr,
             std::vector<tensor_t>& input,
             std::vector<tensor_t>& label,
             size_t batch_size,
             size_t epoch);

    void fit_batch(loss* l_ptr,
                   optimizer* opt_ptr,
                   const tensor_t* input,
                   const tensor_t* label,
                   size_t batch_size);

    void compute(loss* l_ptr,
                 optimizer* opt_ptr,
                 const tensor_t* input,
                 const tensor_t* label,
                 size_t batch_size);

    void newaxis(const tensor_t& in,
                 std::vector<tensor_t>& out);

    void label2vec(const std::vector<size_t>& label,
                   std::vector<tensor_t>& output);

protected:
    mat_t fprop(const mat_t& in);
    std::vector<mat_t> fprop(const std::vector<mat_t>& in);
    std::vector<tensor_t> fprop(const std::vector<tensor_t>& in);

    void bprop(loss* l_ptr,
               optimizer* opt_ptr,
               const std::vector<tensor_t>& net_out,
               const std::vector<tensor_t>& label);

    friend void construct_graph(network<graph>& net,
                                const std::vector<layer*>& input,
                                const std::vector<layer*>& out);

private:
    Net net_;
    std::string network_name_;
};

} // xsdnn

#endif //XSDNN_NETWORK_H
