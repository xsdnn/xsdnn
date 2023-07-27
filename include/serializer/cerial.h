//
// Created by rozhin on 25.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_CERIAL_H
#define XSDNN_CERIAL_H

#include "xs.proto3.pb.h"
#include "../layers/fully_connected.h"
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
    /*
     * Fully Connected
     */
    inline static void serialize(xs::NodeInfo* node, xs::TensorInfo* tensor, const fully_connected* layer) {
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

    inline
    static
    std::shared_ptr<fully_connected> deserialize(const xs::NodeInfo* node,
                                                 const xs::TensorInfo* tensor) {
        size_t in_size = node->attribute(0).i();
        size_t out_size = node->attribute(1).i();
        bool has_bias = node->attribute(2).i();
        std::shared_ptr<fully_connected> l = std::make_shared<fully_connected>(in_size, out_size, has_bias);
        l->load(tensor);
        return l;
    }
};

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

template<typename T>
void load(const xs::NodeInfo* n, const xs::TensorInfo* t, layer* l) {
    cerial::deserialize(n, t, dynamic_cast<T*>(l));
}

void layer_register();

} // xsdnn

#endif //XSDNN_CERIAL_H
