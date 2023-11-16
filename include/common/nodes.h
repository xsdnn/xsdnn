//
// Created by rozhin on 31.03.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_NODES_H
#define XSDNN_NODES_H

#include "node.h"
#include "../layers/layer.h"

namespace xsdnn {

/*
 * Basic class for neural network seq
 */

class nodes {
public:
    nodes();
    virtual ~nodes();

public:
    typedef std::vector<layer*>::iterator iterator;
    typedef std::vector<layer*>::const_iterator const_iterator;

    virtual
    void
    forward(const mat_t& start) = 0;

    virtual
    void
    forward(const tensor_t& start) = 0;

    virtual
    void
    setup(bool reset_weight);

    void
    set_num_threads();

    void save_model(const std::string& filename, const std::string& network_name_);
    void load_model(const std::string& filename);

    bool have_engine_xnnpack();

    size_t size() const;
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    layer* operator[] (size_t index);
    const layer* operator[] (size_t index) const;
    size_t in_data_size() const;
    size_t out_data_size() const;

public:
    size_t user_num_threads_ = 1;

protected:
    void reorder_input(const std::vector<tensor_t> &input,
                       std::vector<tensor_t> &output);

public:
    std::vector<std::shared_ptr<layer>> owner_nodes_; // for r-value impl
    std::vector<layer*> nodes_;
};


class graph : public nodes {
public:
    graph();
    virtual ~graph();

public:
    virtual void forward(const mat_t& start);
    virtual void forward(const tensor_t& start);

    /*
     * Задача метода построить последовательность отсортированных нод
     * для forward и backward проходов нейросети.
     */
    void construct(const std::vector<layer*>& input,
                 const std::vector<layer*>& output);

#ifdef XS_USE_SERIALIZATION
    void save_connections(xs::GraphInfo* Graph);
    void load_connections(xs::GraphInfo* Graph);
#endif

protected:
    void reorder_output(std::vector<tensor_t>& output);

    size_t find_index(const std::vector<node *> &nodes, layer *target);

    friend class network;

private:
    // Возвращает число нод на движке xnnpack
    size_t get_num_xnnpack_backend_engine() const noexcept;

public:
    std::vector<layer*> input_layers_;
    std::vector<layer*> output_layers_;
    std::vector<xsMemoryFormat> mtypes_;
    friend class InfSession;
};

} // xsdnn

#endif //XSDNN_NODES_H
