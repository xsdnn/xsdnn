//
// Created by rozhin on 25.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_CERIAL_H
#define XSDNN_CERIAL_H

#include "xs.proto3.pb.h"
#include "../layers/layers.h"
#include "../utils/macro.h"

namespace xsdnn {

/*
 * Правила сериализации:
 *
 *      1. В комментариях указать название слоя.
 *      2. Сигнатура функции сохранения - void save(node/tensor* n, const layer_type* l);
 *      3. Сигнатура функции загрузки - void load(const node/tensor* n, layer_type* l);
 */

struct cerial {
    template<typename T>
    inline
    std::shared_ptr<T> deserialize(const xs::NodeInfo* node,
                                     const xs::TensorInfo* tensor);
    /*
     * Fully Connected
     */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const fully_connected* layer) {
        node->set_name("fully_connected");
        xs::AttributeInfo* in_size = node->add_attribute();
        xs::AttributeInfo* out_size = node->add_attribute();
        xs::AttributeInfo* has_bias = node->add_attribute();

        in_size->set_name("in_size");
        in_size->set_type(xs::AttributeInfo_AttributeType_INT);
        in_size->set_i(layer->params_.in_size_);

        out_size->set_name("out_size");
        out_size->set_type(xs::AttributeInfo_AttributeType_INT);
        out_size->set_i(layer->params_.out_size_);

        has_bias->set_name("has_bias");
        has_bias->set_type(xs::AttributeInfo_AttributeType_INT);
        has_bias->set_i(layer->params_.has_bias_);

        std::vector<const mat_t*> wb = layer->weights();
        tensor->set_name("w&b fully_connected");
#ifdef XS_USE_DOUBLE
#error NotImplementedYet
#else
        tensor->set_type(xs::TensorInfo_TensorType_FLOAT);
#endif
        layer->save(tensor);
    }

    /*
     * Input
     */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const Input* layer) {
        node->set_name("Input");
        xs::AttributeInfo* C = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* W = node->add_attribute();

        C->set_name("channel");
        C->set_type(xs::AttributeInfo_AttributeType_INT);
        C->set_i(layer->shape_.C);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->shape_.H);

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->shape_.W);
    }

    /*
     * Output
     */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const Output* layer) {
        node->set_name("Output");
        xs::AttributeInfo* C = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* W = node->add_attribute();

        C->set_name("channel");
        C->set_type(xs::AttributeInfo_AttributeType_INT);
        C->set_i(layer->shape_.C);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->shape_.H);

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->shape_.W);
    }

    /*
     * Add
     */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const add* layer) {
        node->set_name("add");
        xs::AttributeInfo* n_input = node->add_attribute();
        xs::AttributeInfo* C = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* W = node->add_attribute();

        n_input->set_name("n_input");
        n_input->set_type(xs::AttributeInfo_AttributeType_INT);
        n_input->set_i(layer->n_input_);

        C->set_name("channel");
        C->set_type(xs::AttributeInfo_AttributeType_INT);
        C->set_i(layer->shape_.C);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->shape_.H);

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->shape_.W);
    }

    /*
     * Relu
     */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const relu* layer) {
        node->set_name("relu");
    }

    /*
    * Abs
    */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const xsdnn::abs* layer) {
        node->set_name("abs");
    }

    /*
    * Acos
    */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const xsdnn::acos* layer) {
        node->set_name("acos");
    }

    /*
    * And
    */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const xsdnn::and_layer* layer) {
        node->set_name("and_layer");
    }

    /*
    * Flatten
    */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const xsdnn::flatten* layer) {
        node->set_name("flatten");
    }

    /*
    * Max Pooling
    */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const xsdnn::max_pooling* layer) {
        node->set_name("max_pooling");
        xs::AttributeInfo* C = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* W = node->add_attribute();
        xs::AttributeInfo* kernel_x = node->add_attribute();
        xs::AttributeInfo* kernel_y = node->add_attribute();
        xs::AttributeInfo* stride_x = node->add_attribute();
        xs::AttributeInfo* stride_y = node->add_attribute();
        xs::AttributeInfo* pad_type = node->add_attribute();

        C->set_name("channel");
        C->set_type(xs::AttributeInfo_AttributeType_INT);
        C->set_i(layer->params_.in_shape_.C);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->params_.in_shape_.H);

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->params_.in_shape_.W);

        kernel_x->set_name("kernel_x");
        kernel_x->set_type(xs::AttributeInfo_AttributeType_INT);
        kernel_x->set_i(layer->params_.kernel_x_);

        kernel_y->set_name("kernel_y");
        kernel_y->set_type(xs::AttributeInfo_AttributeType_INT);
        kernel_y->set_i(layer->params_.kernel_y_);

        stride_x->set_name("stride_x");
        stride_x->set_type(xs::AttributeInfo_AttributeType_INT);
        stride_x->set_i(layer->params_.stride_x_);

        stride_y->set_name("stride_y");
        stride_y->set_type(xs::AttributeInfo_AttributeType_INT);
        stride_y->set_i(layer->params_.stride_y_);

        pad_type->set_name("pad_type");
        pad_type->set_type(xs::AttributeInfo_AttributeType_STRING);
        pad_type->set_name((layer->params_.pad_type_ == padding_mode::same) ? "same" : "valid");
    }

    /*
    * Global Average Pooling
    */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const xsdnn::global_average_pooling* layer) {
        node->set_name("global_average_pooling");
        xs::AttributeInfo* C = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* W = node->add_attribute();

        C->set_name("channel");
        C->set_type(xs::AttributeInfo_AttributeType_INT);
        C->set_i(layer->params_.in_shape_.C);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->params_.in_shape_.H);

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->params_.in_shape_.W);
    }

    /*
    * Reshape
    */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const xsdnn::reshape* layer) {
        node->set_name("reshape");
        xs::AttributeInfo* C = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* W = node->add_attribute();

        C->set_name("channel");
        C->set_type(xs::AttributeInfo_AttributeType_INT);
        C->set_i(layer->out_shape_.C);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->out_shape_.H);

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->out_shape_.W);
    }

};

    template<>
    inline
    std::shared_ptr<fully_connected> cerial::deserialize(const xs::NodeInfo* node,
                                                 const xs::TensorInfo* tensor) {
        size_t in_size = node->attribute(0).i();
        size_t out_size = node->attribute(1).i();
        bool has_bias = node->attribute(2).i();
        std::shared_ptr<fully_connected> l = std::make_shared<fully_connected>(in_size, out_size, has_bias);
        l->load(tensor);
        return l;
    }

    template<>
    inline
    std::shared_ptr<Input> cerial::deserialize(const xs::NodeInfo* node,
                                       const xs::TensorInfo* tensor) {
        size_t C = node->attribute(0).i();
        size_t H = node->attribute(1).i();
        size_t W = node->attribute(2).i();
        std::shared_ptr<Input> l = std::make_shared<Input>(shape3d(C, H, W));
        return l;
    }

    template<>
    inline
    std::shared_ptr<Output> cerial::deserialize(const xs::NodeInfo* node,
                                               const xs::TensorInfo* tensor) {
        size_t C = node->attribute(0).i();
        size_t H = node->attribute(1).i();
        size_t W = node->attribute(2).i();
        std::shared_ptr<Output> l = std::make_shared<Output>(shape3d(C, H, W));
        return l;
    }

    template<>
    inline
    std::shared_ptr<add> cerial::deserialize(const xs::NodeInfo* node,
                                               const xs::TensorInfo* tensor) {
        size_t n_input = node->attribute(0).i();
        size_t C = node->attribute(1).i();
        size_t H = node->attribute(2).i();
        size_t W = node->attribute(3).i();
        std::shared_ptr<add> l = std::make_shared<add>(n_input, shape3d(C, H, W));
        return l;
    }

    template<>
    inline
    std::shared_ptr<relu> cerial::deserialize(const xs::NodeInfo* node,
                                             const xs::TensorInfo* tensor) {
        std::shared_ptr<relu> l = std::make_shared<relu>();
        return l;
    }

    template<>
    inline
    std::shared_ptr<xsdnn::abs> cerial::deserialize(const xs::NodeInfo* node,
                                              const xs::TensorInfo* tensor) {
        std::shared_ptr<xsdnn::abs> l = std::make_shared<xsdnn::abs>();
        return l;
    }

    template<>
    inline
    std::shared_ptr<xsdnn::acos> cerial::deserialize(const xs::NodeInfo* node,
                                                    const xs::TensorInfo* tensor) {
        std::shared_ptr<xsdnn::acos> l = std::make_shared<xsdnn::acos>();
        return l;
    }

    template<>
    inline
    std::shared_ptr<xsdnn::and_layer> cerial::deserialize(const xs::NodeInfo* node,
                                                     const xs::TensorInfo* tensor) {
        std::shared_ptr<xsdnn::and_layer> l = std::make_shared<xsdnn::and_layer>();
        return l;
    }

    template<>
    inline
    std::shared_ptr<xsdnn::flatten> cerial::deserialize(const xs::NodeInfo* node,
                                                          const xs::TensorInfo* tensor) {
        std::shared_ptr<xsdnn::flatten> l = std::make_shared<xsdnn::flatten>();
        return l;
    }

    template<>
    inline
    std::shared_ptr<xsdnn::max_pooling> cerial::deserialize(const xs::NodeInfo *node,
                                                            const xs::TensorInfo *tensor) {
        size_t C = node->attribute(0).i();
        size_t H = node->attribute(1).i();
        size_t W = node->attribute(2).i();
        size_t kernel_x = node->attribute(3).i();
        size_t kernel_y = node->attribute(4).i();
        size_t stride_x = node->attribute(5).i();
        size_t stride_y = node->attribute(6).i();
        padding_mode pad_type = node->attribute(0).s() == "same"
                            ? padding_mode::same : padding_mode::valid;

        std::shared_ptr<xsdnn::max_pooling> l = std::make_shared<xsdnn::max_pooling>(
                shape3d(C, H, W), kernel_x, kernel_y, stride_x, stride_y, pad_type
                );
        return l;
    }

    template<>
    inline
    std::shared_ptr<xsdnn::global_average_pooling> cerial::deserialize(const xs::NodeInfo *node,
                                                            const xs::TensorInfo *tensor) {
        size_t C = node->attribute(0).i();
        size_t H = node->attribute(1).i();
        size_t W = node->attribute(2).i();
        std::shared_ptr<global_average_pooling> l = std::make_shared<global_average_pooling>(shape3d(C, H, W));
        return l;
    }

    template<>
    inline
    std::shared_ptr<xsdnn::reshape> cerial::deserialize(const xs::NodeInfo *node,
                                                           const xs::TensorInfo *tensor) {
        size_t C = node->attribute(0).i();
        size_t H = node->attribute(1).i();
        size_t W = node->attribute(2).i();
        std::shared_ptr<reshape> l = std::make_shared<reshape>(shape3d(C, H, W));
        return l;
    }




















class serializer {
public:
    static serializer& get_instance() {
        static serializer instance;
        return instance;
    }

    void register_saver(std::string layer_name,
                        std::function<void(xs::NodeInfo*, xs::TensorInfo*, const layer*)> func) {
        saver_[layer_name] = func;
    }

    void register_loader(std::string layer_name,
                         std::function<void(const xs::NodeInfo*, const xs::TensorInfo*)> func) {
        loader_[layer_name] = func;
    }

    void save(xs::NodeInfo* node, xs::TensorInfo* tensor, const layer* layer) {
        std::string layer_typename = layer->layer_type();
        if (node->IsInitialized() && tensor->IsInitialized() && layer != nullptr) {
            saver_[layer_typename](node, tensor, layer);
        }
    }

    void load(const xs::NodeInfo* node, const xs::TensorInfo* tensor,
              std::vector<std::shared_ptr<layer>>& owner_nodes);

private:
    std::map<std::string, std::function<void(xs::NodeInfo*, xs::TensorInfo*, const layer*)>> saver_;
    std::map<std::string, std::function<void(const xs::NodeInfo*, const xs::TensorInfo*)>> loader_;
};

template<typename T>
void save(xs::NodeInfo* n, xs::TensorInfo* t, const layer* l) {
    cerial::serialize(n, t, dynamic_cast<const T*>(l));
}

void layer_register();

} // xsdnn

#endif //XSDNN_CERIAL_H
