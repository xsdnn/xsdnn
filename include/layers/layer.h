//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_LAYER_H
#define XSDNN_LAYER_H

#include <common/node.h>
#include <utils/weight_init.h>
#include <optimizers/optimizer_base.h>
#include <core/backend.h>

namespace xsdnn {

class layer : public node {
public:
    layer(const std::vector<tensor_type>& in_type,
          const std::vector<tensor_type>& out_type);
    virtual ~layer();


    void set_parallelize(bool parallelize);
    void set_num_threads(size_t num_threads);
    void set_backend(core::backend_t engine);
    core::backend_t engine() const;

    bool parallelize() const;
    size_t in_concept() const;
    size_t out_concept() const;
    bool trainable() const;

    size_t in_data_size() const;
    size_t out_data_size() const;
    std::vector<shape3d> in_data_shape() const;
    std::vector<shape3d> out_data_shape() const;

    std::vector<const mat_t*> weights() const;
    std::vector<mat_t*> weights();
    std::vector<tensor_t*> weights_grads();

    std::vector<edgeptr_t> inputs();
    std::vector<edgeptr_t> outputs();
    std::vector<edgeptr_t> outputs() const;

    void set_in_data(const std::vector<tensor_t>& data);
    void set_out_grads(const std::vector<tensor_t>& grad);
    void set_trainable(bool trainable);

    std::vector<tensor_t> output() const;

    std::vector<tensor_type> in_types() const;
    std::vector<tensor_type> out_types() const;

    template<typename WeightInit>
    void weight_init(const WeightInit& f);

    template<typename BiasInit>
    void bias_init(const BiasInit& f);

    void
    post_update() {}

    virtual
    std::vector<shape3d>
    in_shape() const = 0;

    virtual
    std::vector<shape3d>
    out_shape() const = 0;

    virtual
    std::string
    layer_type() const = 0;

    virtual
    size_t
    fan_in_size() const = 0;

    virtual
    size_t
    fan_out_size() const = 0;

    virtual
    void
    set_in_shape(const shape3d in_shape);


    virtual std::pair<mm_scalar, mm_scalar> out_value_range() const;

    /*
     * Forward \ backward propagation
     */

    virtual
    void
    forward_propagation(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>& out_data) = 0;

    virtual
    void
    back_propagation(const std::vector<tensor_t*>& in_data,
                     const std::vector<tensor_t*>& out_data,
                     std::vector<tensor_t*>&       out_grad,
                     std::vector<tensor_t*>&       in_grad) = 0;


    void forward();

    void backward();

    void setup(bool reset_weight);

    void init_weight();

    void clear_grads();

    void update_weight(optimizer* opt);

    virtual
    void set_sample_count(size_t sample_count);

public:
    friend void connection_mismatch(const layer& from,
                                    const layer& to);

private:
    void alloc_input(size_t i) const;
    void alloc_output(size_t i) const;
    edgeptr_t ith_in_node(size_t i);
    edgeptr_t ith_out_node(size_t i);
    mat_t* get_weight_data(size_t i);
    const mat_t* get_weight_data(size_t i) const;

protected:
    bool initialized_;
    bool parallelize_;
    size_t num_threads_;
    size_t in_concept_;
    size_t out_concept_;
    std::vector<tensor_type> in_type_;
    std::vector<tensor_type> out_type_;

private:
    bool trainable_;
    mat_t weight_diff_helper_;
    core::backend_t engine_;
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    std::vector<tensor_t*> fwd_in_data;
    std::vector<tensor_t*> fwd_out_data;
    std::vector<tensor_t*> bwd_in_data;
    std::vector<tensor_t*> bwd_in_grad;
    std::vector<tensor_t*> bwd_out_data;
    std::vector<tensor_t*> bwd_out_grad;

    friend class GradChecker;
};

void connect(layer* last_node,
                     layer* next_node,
                     size_t last_node_data_concept_idx,
                     size_t next_node_data_concept_idx);

void connection_mismatch(const layer& from, const layer& to);

} // xsdnn

#endif //XSDNN_LAYER_H
