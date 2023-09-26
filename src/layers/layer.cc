//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/layer.h>
#include <utils/macro.h>

std::string convert_ttype_to_string(xsdnn::tensor_type ttype) {
    switch (ttype) {
        case xsdnn::tensor_type::data:
            return std::string("data");
        case xsdnn::tensor_type::weight:
            return std::string("weight");
        case xsdnn::tensor_type::bias:
            return std::string("bias");
        default:
            throw xsdnn::xs_error("Unsupported tensor type");
    }
}

namespace xsdnn {

    layer::layer(const std::vector<TypeHolder> &in_type,
                 const std::vector<TypeHolder> &out_type)
        : node(in_type.size(), out_type.size()),
                initialized_(false),
                parallelize_(true),
                in_concept_(in_type.size()),
                out_concept_(out_type.size()),
                in_type_(in_type),
                out_type_(out_type) {
            weight_init_ = std::make_shared<weight_init::xavier>();
            bias_init_ = std::make_shared<weight_init::xavier>();
            trainable_ = true;
    }

    layer::~layer() {}

    void layer::set_parallelize(bool parallelize) {
        parallelize_ = parallelize;
        if (!parallelize_) {
            num_threads_ = 1;
        }
    }

    void layer::set_num_threads(size_t num_threads) {
        num_threads_ = num_threads;
        if (num_threads_ == 1) {
            parallelize_ = false;
        }
    }

    void layer::set_backend(core::backend_t engine) {
        engine_ = engine;
    }

    bool layer::parallelize() const {
        return parallelize_;
    }

    core::backend_t layer::engine() const {
        return engine_;
    }

    size_t layer::in_concept() const {
        return in_concept_;
    }

    size_t layer::out_concept() const {
        return out_concept_;
    }

    size_t layer::in_data_size() const {
        size_t result_size = 0;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (in_type_[i].concept_type() == tensor_type::data) {
                result_size += in_shape()[i].size();
            }
        }
        return result_size;
    }

    size_t layer::out_data_size() const {
        size_t result_size = 0;
        for (size_t i = 0; i < out_concept_; ++i) {
            if (out_type_[i].concept_type() == tensor_type::data) {
                result_size += out_shape()[i].size();
            }
        }
        return result_size;
    }

    std::vector<shape3d> layer::in_data_shape() const {
        std::vector<shape3d> ids;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (in_type_[i].concept_type() == tensor_type::data) {
                ids.push_back(in_shape()[i]);
            }
        }
        return ids;
    }

    std::vector<shape3d> layer::out_data_shape() const {
        std::vector<shape3d> ids;
        for (size_t i = 0; i < out_concept_; ++i) {
            if (out_type_[i].concept_type() == tensor_type::data) {
                ids.push_back(out_shape()[i]);
            }
        }
        return ids;
    }

    void layer::set_in_shape(const xsdnn::shape3d in_shape) {
        throw xs_error("You can't set input shape. Sorry ^(");
    }

    std::vector<const tensor_t *> layer::weights() const {
        std::vector<const tensor_t *> v;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (is_trainable_concept(in_type_[i].concept_type())) {
                v.push_back(get_weight_data(i));
            }
        }
        return v;
    }

    std::vector<tensor_t *> layer::weights() {
        std::vector<tensor_t *> v;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (is_trainable_concept(in_type_[i].concept_type())) {
                v.push_back(get_weight_data(i));
            }
        }
        return v;
    }

    std::vector<edgeptr_t> layer::inputs() {
        std::vector<edgeptr_t> nodes;
        for (size_t i = 0; i < in_concept_; ++i) {
            nodes.push_back(ith_in_node(i));
        }
        return nodes;
    }

    std::vector<edgeptr_t> layer::outputs() {
        std::vector<edgeptr_t> nodes;
        for (size_t i = 0; i < out_concept_; ++i) {
            nodes.push_back(ith_out_node(i));
        }
        return nodes;
    }

    std::vector<edgeptr_t> layer::outputs() const {
        std::vector<edgeptr_t> nodes;
        for (size_t i = 0; i < out_concept_; ++i) {
            nodes.push_back(const_cast<layer*>(this)->ith_out_node(i));
        }
        return nodes;
    }

    void layer::set_in_data(const std::vector<BTensor> &data) {
        size_t data_idx = 0;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (in_type_[i].concept_type() != tensor_type::data) continue;
            assert(data_idx < data.size());
            *ith_in_node(i)->get_data() = data[data_idx++];
        }
    }

    std::vector<BTensor> layer::output() const {
        std::vector<BTensor> out;
        for (size_t i = 0; i < out_concept_; ++i) {
            if (out_type_[i].concept_type() == tensor_type::data) {
                out.push_back(*(const_cast<layer*>(this))
                                      ->ith_out_node(i)->get_data());
            }
        }
        return out;
    }

    std::vector<TypeHolder> layer::in_types() const {
        return in_type_;
    }

    std::vector<TypeHolder> layer::out_types() const {
        return out_type_;
    }

    void layer::set_trainable(bool trainable) {
        trainable_ = trainable;
    }

    bool layer::trainable() const {
        return trainable_;
    }

    void layer::forward() {
        fwd_in_data.reserve(in_concept_);
        fwd_out_data.reserve(out_concept_);

        for (size_t i = 0; i < in_concept_; ++i) {
            fwd_in_data.emplace_back(ith_in_node(i)->get_data());
        }

        // Find the tensor_type::data index
        int32_t data_idx = -1;
        for (int32_t i = 0; i < (int32_t) in_concept_; ++i) {
            if (in_type_[i].concept_type() == tensor_type::data) data_idx = i;
        }

        if (data_idx == -1) {
            throw xs_error("Not found \'data\' tensor type.");
        } else {
            set_sample_count(fwd_in_data[(size_t) data_idx]->size());
        }

        for (size_t i = 0; i < out_concept_; ++i) {
            fwd_out_data.emplace_back(ith_out_node(i)->get_data());
        }

        forward_propagation(fwd_in_data, fwd_out_data);
    }

    void layer::setup(bool reset_weight) {
        if (in_shape().size() != in_concept_ ||
            out_shape().size() != out_concept_) {
            xs_error("Connection mismatch at setup layer.");
        }

        for (size_t i = 0; i < out_concept_; ++i) {
            if (!next_[i]) {
                next_[i] = std::make_shared<edge>(this,
                                                  out_shape()[i],
                                                  out_type_[i].concept_type(),
                                                  out_type_[i].tensor_dtype());
            }
        }
    }

    void layer::set_sample_count(size_t sample_count) {
        auto resize = [sample_count](BTensor* tensor) {
            tensor->resize(sample_count, (*tensor)[0]);
        };

        for (size_t i = 0; i < in_concept_; ++i) {
            if (!is_trainable_concept(in_type_[i].concept_type())) {
                resize(ith_in_node(i)->get_data());
            }
        }

        for (size_t i = 0; i < out_concept_; ++i) {
            if (!is_trainable_concept(out_type_[i].concept_type())) {
                resize(ith_out_node(i)->get_data());
            }
        }
    }

    void layer::alloc_input(size_t i) const {
        prev_[i] = std::make_shared<edge>(nullptr,
                                          in_shape()[i], in_type_[i].concept_type(), in_type_[i].tensor_dtype());
    }

    void layer::alloc_output(size_t i) const {
        next_[i] = std::make_shared<edge>((layer*) this,
                                          out_shape()[i], out_type_[i].concept_type(), out_type_[i].tensor_dtype());
    }

    edgeptr_t layer::ith_in_node(size_t i) {
        if (!prev_[i]) alloc_input(i);
        return prev_[i];
    }

    edgeptr_t layer::ith_out_node(size_t i) {
        if (!next_[i]) alloc_output(i);
        return next_[i];
    }

    tensor_t *layer::get_weight_data(size_t i) {
        assert(is_trainable_concept(in_type_[i].concept_type()));
        return &(*ith_in_node(i)->get_data())[0];
    }

    const tensor_t *layer::get_weight_data(size_t i) const {
        assert(is_trainable_concept(in_type_[i].concept_type()));
        return &(*const_cast<layer*>(this)->ith_in_node(i)->get_data())[0];
    }

    std::pair<mm_scalar, mm_scalar> layer::out_value_range() const {
        return { mm_scalar(0.0f), mm_scalar(1.0f) };
    }

    void connect(layer* last_node,
                        layer* next_node,
                        size_t last_node_data_concept_idx = 0,
                        size_t next_node_data_concept_idx = 0) {
        auto out_shape = last_node->out_shape()[last_node_data_concept_idx];
        auto in_shape = next_node->in_shape()[next_node_data_concept_idx];

        last_node->setup(false);

        if (in_shape.size() == 0 /* eq. this activation */) {
            next_node->set_in_shape(out_shape);
            in_shape = out_shape;
        }

        if (out_shape.size() != in_shape.size()) {
            connection_mismatch(*last_node, *next_node);
        }

        if (!last_node->next_[last_node_data_concept_idx]) {
            xs_error("Not alloc \"data\" on last node");
        }

        next_node->prev_[next_node_data_concept_idx] = last_node->next_[last_node_data_concept_idx];
        next_node->prev_[next_node_data_concept_idx]->add_next_node(next_node);
    }

    void connection_mismatch(const layer& from, const layer& to) {
        std::ostringstream io;
        io << "\x1B[31m" << "Critical Error! Layer mismatch!" << std::endl;
        io << "\x1B[33m" << "Layer's [N, N + 1]: " << from.layer_type() << " -> " << to.layer_type() << std::endl;

        io << "\x1B[33m" << "N: in=" << from.in_data_size() << ", out=" << from.out_data_size()
            << " | in_shape=" << from.in_shape() << ", out_shape=" << from.out_shape() << std::endl;

        io << "\x1B[33m" << "N+1: in=" << to.in_data_size() << ", out=" << to.out_data_size()
           << " | in_shape=" << to.in_shape() << ", out_shape=" << to.out_shape();

        std::string message = io.str();
        throw xs_error(message.c_str());
    }

    std::vector<TypeHolder> get_typed_holder(bool has_bias, std::vector<XsDtype> dtype) {
        if (has_bias) {
            assert(dtype.size() == 3);
            return {TypeHolder(tensor_type::data, dtype[0]),
                    TypeHolder(tensor_type::weight, dtype[1]),
                    TypeHolder(tensor_type::bias, dtype[2])};
        } else {
            assert(dtype.size() == 2);
            return {TypeHolder(tensor_type::data, dtype[0]),
                    TypeHolder(tensor_type::weight, dtype[1])};
        }
    }

} // xsdnn