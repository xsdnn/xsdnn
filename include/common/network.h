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

class network {
public:
    typedef std::vector<layer*>::iterator iterator;
    typedef std::vector<layer*>::const_iterator const_iterator;

    network() = default;
    network(const network&) = default;
    network& operator=(const network&) = default;
    ~network() = default;

    template<typename L>
    network& operator<<(L &&layer);

public:
    void init_weight();
    void set_num_threads(size_t num_threads) noexcept;
    bool empty() const;

    mat_t predict(const mat_t& in);
    tensor_t predict(const tensor_t& in);
    std::vector<tensor_t> predict(const std::vector<tensor_t>& in);

    void train(loss* loss,
               optimizer* opt,
               const tensor_t& input,
               const std::vector<size_t>& label,
               size_t batch_size,
               size_t epoch);


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

private:
    sequential net_;
};

} // xsdnn

#endif //XSDNN_NETWORK_H
