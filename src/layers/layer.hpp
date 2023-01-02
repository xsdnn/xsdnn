//
// Created by Andrei R. on 30.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_LAYER_HPP
#define XSDNN_LAYER_HPP

namespace xsdnn {

class layer : public node {
public:
    layer(const std::vector<tensor_type>& in_type,
          const std::vector<tensor_type>& out_type) :
            node(in_type.size(), out_type.size()),
            initialized_(false),
            in_concept_(in_type.size()),
            out_concept_(out_type.size()),
            in_type_(in_type),
            out_type_(out_type),
            trainable_(true) {
        weight_init_ = std::make_shared<weight_init::xavier>();
        bias_init_ = std::make_shared<weight_init::constant>();
        trainable_ = true;
    }

    /*
     * getters
     */
    size_t in_concept() const {
        return in_concept_;
    }

    size_t out_concept() const {
        return out_concept_;
    }

    bool trainable() const {
        return trainable_;
    }

    bool initialized() const {
        return initialized_;
    }

    std::vector<tensor_type> in_type() const {
        return in_type_;
    }

    std::vector<tensor_type> out_type() const {
        return out_type_;
    }

    virtual std::vector<shape3d> in_shape() const = 0;
    virtual std::vector<shape3d> out_shape() const = 0;

protected:
    /*Indicate layer parameters initializing*/
    bool initialized_;

    /*Count of input/output concept*/
    size_t in_concept_;
    size_t out_concept_;

    /*Container contains input/output concepts. These size equal in_concept/out_concept*/
    std::vector<tensor_type> in_type_;
    std::vector<tensor_type> out_type_;

    /*Type of backeng engine*/
    core::backend_t backend_;

private:
    /*Need we update trainable concept for this layer?*/
    bool trainable_;

    /*Weights&bias concept initializing*/
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    /*Holders for forward and backward propagation*/
    std::vector<Tensor_3D *> fwd_in_data_;
    std::vector<Tensor_3D *> fwd_out_data_;
    std::vector<Tensor_3D *> bwd_in_data_;
    std::vector<Tensor_3D *> bwd_in_grad_;
    std::vector<Tensor_3D *> bwd_out_data_;
    std::vector<Tensor_3D *> bwd_out_grad_;

private:
    /*Allocating memory for input concept*/
    void alloc_input(size_t i) const {
        prev_[i] = std::make_shared<edge>(nullptr, in_shape()[i], in_type_[i]);
    }

    /*Allocating memory for output concept*/
    void alloc_output(size_t i) const {
        next_[i] = std::make_shared<edge>(const_cast<layer *>(this), out_shape()[i], out_type_[i]);
    }

    /*Return edge pointer to ith input concept*/
    edgeptr_t ith_in_node(size_t i) {
        if (!prev_[i]) {
            alloc_input(i);
        }
        return prev_[i];
    }

    /*Return edge pointer to ith output concept*/
    edgeptr_t ith_out_node(size_t i) {
        if (!next_[i]) {
            alloc_output(i);
        }
        return next_[i];
    }
};


} // xsdnn

#endif //XSDNN_LAYER_HPP
