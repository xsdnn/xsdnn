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
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const input* layer) {
        node->set_name("input");
        xs::AttributeInfo* W = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* D = node->add_attribute();

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->shape_.W);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->shape_.H);

        D->set_name("depth");
        D->set_type(xs::AttributeInfo_AttributeType_INT);
        D->set_i(layer->shape_.C);
    }

    /*
     * Output
     */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const output* layer) {
        node->set_name("output");
        xs::AttributeInfo* W = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* D = node->add_attribute();

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->shape_.W);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->shape_.H);

        D->set_name("depth");
        D->set_type(xs::AttributeInfo_AttributeType_INT);
        D->set_i(layer->shape_.C);
    }

    /*
     * Add
     */
    inline
    static
    void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const add* layer) {
        node->set_name("add");
        xs::AttributeInfo* n_input = node->add_attribute();
        xs::AttributeInfo* W = node->add_attribute();
        xs::AttributeInfo* H = node->add_attribute();
        xs::AttributeInfo* D = node->add_attribute();

        n_input->set_name("n_input");
        n_input->set_type(xs::AttributeInfo_AttributeType_INT);
        n_input->set_i(layer->n_input_);

        W->set_name("width");
        W->set_type(xs::AttributeInfo_AttributeType_INT);
        W->set_i(layer->shape_.W);

        H->set_name("height");
        H->set_type(xs::AttributeInfo_AttributeType_INT);
        H->set_i(layer->shape_.H);

        D->set_name("depth");
        D->set_type(xs::AttributeInfo_AttributeType_INT);
        D->set_i(layer->shape_.C);
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
    std::shared_ptr<input> cerial::deserialize(const xs::NodeInfo* node,
                                       const xs::TensorInfo* tensor) {
        size_t W = node->attribute(0).i();
        size_t H = node->attribute(1).i();
        size_t D = node->attribute(2).i();
        std::shared_ptr<input> l = std::make_shared<input>(shape3d(W, H, D));
        return l;
    }

    template<>
    inline
    std::shared_ptr<output> cerial::deserialize(const xs::NodeInfo* node,
                                               const xs::TensorInfo* tensor) {
        size_t W = node->attribute(0).i();
        size_t H = node->attribute(1).i();
        size_t D = node->attribute(2).i();
        std::shared_ptr<output> l = std::make_shared<output>(shape3d(W, H, D));
        return l;
    }

    template<>
    inline
    std::shared_ptr<add> cerial::deserialize(const xs::NodeInfo* node,
                                               const xs::TensorInfo* tensor) {
        size_t n_input = node->attribute(0).i();
        size_t W = node->attribute(1).i();
        size_t H = node->attribute(2).i();
        size_t D = node->attribute(3).i();
        std::shared_ptr<add> l = std::make_shared<add>(n_input, shape3d(W, H, D));
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
