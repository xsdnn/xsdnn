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

    virtual ~layer() = default;

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

    /*Holders for describing in / out concept size*/
    virtual std::vector<shape3d> in_shape() const = 0;
    virtual std::vector<shape3d> out_shape() const = 0;

    /*Unique  type of this layer*/
    virtual std::string layer_type() const = 0;

    /*Num features at input & output stages*/
    virtual Eigen::DenseIndex fan_in_size() const = 0;
    virtual Eigen::DenseIndex fan_out_size() const = 0;

    /*
     * General method, where calling kernel forward algorithm for these layer
     */
    virtual void forward_propagation(const std::vector<Tensor_3D *>& in_data,
                                     std::vector<Tensor_3D *>& out_data) = 0;

    /*
     * General method, where calling kernel backward algorithm for these layer
     */
    virtual void backward_propagation(const std::vector<Tensor_3D *>& in_data,
                                      const std::vector<Tensor_3D *>& out_data,
                                      std::vector<Tensor_3D *>& in_grad,
                                      std::vector<Tensor_3D *>& out_grad) = 0;

    void set_in_data(const Tensor_4D& in_data /*// TODO: можно ли получать 3D ?*/) {
        DNN_UNUSED_PARAMETER(in_data);
        for (size_t i = 0; i < in_concept_; ++i) {
            if (in_type_[i] != tensor_type::data) continue;
            Tensor_3D& dst_data = ith_in_node(i)->get_data();
            DNN_UNUSED_PARAMETER(dst_data);
            // TODO: единственный вариант, как на данном этапе не делать увеличение ранга тензора, но учесть кол-во
            // элементов в батче - resize (!batch_size! * channel, height, width)
            // FIXME: подумать, нужно ли после расчетов - возвращать данные в исходный 3D формат - предварительно да.
            // TODO: подумать, как коннектятся слои между собой
            //  :: https://github.com/tiny-dnn/tiny-dnn/blob/c0f576f5cb7b35893f62127cb7aec18f77a3bcc5/tiny_dnn/layers/layer.h#L863
        }
    }

    void forward() {
        // TODO: implement this place...
        // How we move from 3D to 4D Tensor
    }

    /*
     * Main method for w&b allocating
     */
    void setup(bool reset_wb) {
        if (in_shape().size() != in_concept_ ||
            out_shape().size() != out_concept_) {
            xs_error("Connection mismatch. You make programming mistake!");
        }

        /*
         * Allocating data for output concept
         */
        for (size_t i = 0; i < out_concept_; ++i) {
            if (!next_[i]) {
                alloc_output(i);
            }
        }

        /*
         * Allocating inputs w&b
         */
        if (!initialized_ || reset_wb) {
            init_weight();
        }
    }

    void init_weight() {
        if (!trainable_) {
            initialized_ = true;
            return;
        }

        for (size_t i = 0; i < in_concept_; ++i) {
            switch (in_type_[i]) {
                case tensor_type::weight: {
                    Tensor_3D *w = get_weight_data(i);
                    weight_init_->fill(w->data(), w->size(), fan_in_size(), fan_out_size());
                    break;
                }

                case tensor_type::bias: {
                    Tensor_3D *b = get_weight_data(i);
                    bias_init_->fill(b->data(), b->size(), fan_in_size(), fan_out_size());
                    break;
                }

                default:
                    break;
            }
        }
    }

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

    /*For serialization task*/
    friend struct cerial;

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

    Tensor_3D* get_weight_data(size_t i) {
        assert(is_trainable_concept(in_type_[i]));
        return &ith_in_node(i)->get_data();
    }

    /*
     * Connecting prev / next layer in a sequential
     */
    virtual void connect(layer *prev,
                         layer *next,
                         size_t prev_data_concept_idx = 0,
                         size_t next_data_concept_idx = 0) override{
        auto prev_shape = prev->out_shape()[prev_data_concept_idx];
        auto next_shape = next->in_shape()[next_data_concept_idx];

        if (prev_shape.size() != next_shape.size()) {
            // TODO: add connection mismatch method
            xs_error("Connection mismatch. You make programming mistake!");
        }

        if (!prev->next_[prev_data_concept_idx]) {
            xs_error("Output data on previous layer must be not nullptr");
        }

        next->prev_[next_data_concept_idx] = prev->next_[prev_data_concept_idx];
        // next->prev_[next_data_concept_idx]->add_next_node(next);
    }
};


} // xsdnn

#endif //XSDNN_LAYER_HPP
