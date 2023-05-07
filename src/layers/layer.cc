//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "layer.h"

namespace xsdnn {

    layer::layer(const std::vector<tensor_type> &in_type, const std::vector<tensor_type> &out_type)
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
            if (in_type_[i] == tensor_type::data) {
                result_size += in_shape()[i].size();
            }
        }
        return result_size;
    }

    size_t layer::out_data_size() const {
        size_t result_size = 0;
        for (size_t i = 0; i < out_concept_; ++i) {
            if (out_type_[i] == tensor_type::data) {
                result_size += out_shape()[i].size();
            }
        }
        return result_size;
    }

    std::vector<shape3d> layer::in_data_shape() const {
        std::vector<shape3d> ids;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (in_type_[i] == tensor_type::data) {
                ids.push_back(in_shape()[i]);
            }
        }
        return ids;
    }

    std::vector<shape3d> layer::out_data_shape() const {
        std::vector<shape3d> ids;
        for (size_t i = 0; i < out_concept_; ++i) {
            if (out_type_[i] == tensor_type::data) {
                ids.push_back(out_shape()[i]);
            }
        }
        return ids;
    }

    std::vector<const mat_t *> layer::weights() const {
        std::vector<const mat_t *> v;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (is_trainable_concept(in_type_[i])) {
                v.push_back(get_weight_data(i));
            }
        }
        return v;
    }

    std::vector<mat_t *> layer::weights() {
        std::vector<mat_t *> v;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (is_trainable_concept(in_type_[i])) {
                v.push_back(get_weight_data(i));
            }
        }
        return v;
    }

    std::vector<tensor_t *> layer::weights_grads() {
        std::vector<tensor_t*> v;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (is_trainable_concept(in_type_[i])) {
                v.push_back(ith_in_node(i)->get_gradient());
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

    void layer::set_out_grads(const std::vector<tensor_t> &grad) {
        size_t grad_idx = 0;
        for (size_t i = 0; i < out_concept_; ++i) {
            if (out_type_[i] != tensor_type::data) continue;
            assert(grad_idx < grad.size());
            *ith_out_node(i)->get_gradient() = grad[grad_idx++];
        }
    }

    void layer::set_in_data(const std::vector<tensor_t> &data) {
        size_t data_idx = 0;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (in_type_[i] != tensor_type::data) continue;
            assert(data_idx < data.size());
            *ith_in_node(i)->get_data() = data[data_idx++];
        }
    }

    std::vector<tensor_t> layer::output() const {
        std::vector<tensor_t> out;
        for (size_t i = 0; i < out_concept_; ++i) {
            if (out_type_[i] == tensor_type::data) {
                out.push_back(*(const_cast<layer*>(this))
                                      ->ith_out_node(i)->get_data());
            }
        }
        return out;
    }

    std::vector<tensor_type> layer::in_types() const {
        return in_type_;
    }

    std::vector<tensor_type> layer::out_types() const {
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
            if (in_type_[i] == tensor_type::data) data_idx = i;
        }

        if (data_idx == -1) {
            throw xs_error("Not found \'data\' tensor type.");
        } else {
            set_sample_count(fwd_in_data[(size_t) data_idx]->size());
        }

        for (size_t i = 0; i < out_concept_; ++i) {
            fwd_out_data.emplace_back(ith_out_node(i)->get_data());
            ith_out_node(i)->clear_grads();
        }

        forward_propagation(fwd_in_data, fwd_out_data);
    }

    void layer::backward() {
        bwd_in_data.reserve(in_concept_);
        bwd_in_grad.reserve(in_concept_);
        bwd_out_data.reserve(out_concept_);
        bwd_out_grad.reserve(out_concept_);

        for (size_t i = 0; i < in_concept_; ++i) {
            const auto& nd = ith_in_node(i);
            bwd_in_data.emplace_back(nd->get_data());
            bwd_in_grad.emplace_back(nd->get_gradient());
        }

        for (size_t i = 0; i < out_concept_; ++i) {
            const auto& nd = ith_out_node(i);
            bwd_out_data.emplace_back(nd->get_data());
            bwd_out_grad.emplace_back(nd->get_gradient());
        }

        back_propagation(bwd_in_data, bwd_out_data, bwd_out_grad, bwd_in_grad);
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
                                                  out_type_[i]);
            }
        }

        if (reset_weight || !initialized_) {
            init_weight();
        }
    }

    void layer::init_weight() {
        if (!trainable_) {
            initialized_ = true;
            return;
        }

        for (size_t i = 0; i < in_concept_; ++i) {
            if (in_type_[i] == tensor_type::weight) {
                auto* w = get_weight_data(i);
                weight_init_->fill(
                        w->data(),
                        w->size(),
                        fan_in_size(), fan_out_size());
            } else if (in_type_[i] == tensor_type::bias) {
                auto* b = get_weight_data(i);
                bias_init_->fill(
                        b->data(),
                        b->size(),
                        fan_in_size(), fan_out_size());
            }
        }

        initialized_ = true;
    }

    void layer::clear_grads() {
        for(size_t i = 0; i < in_concept_; ++i) {
            ith_in_node(i)->clear_grads();
        }
    }

    void layer::update_weight(optimizer *opt) {
        auto& dw = weight_diff_helper_;
        for (size_t i = 0; i < in_concept_; ++i) {
            if (trainable_ && is_trainable_concept(in_type_[i])) {
                mat_t& w = *get_weight_data(i);
                ith_in_node(i)->accumulate_grads(&dw);
                mm_scalar rcp_batch_size =
                        (mm_scalar) 1.0 / (mm_scalar) ith_in_node(i)->get_data()->size();

                for (size_t j = 0; j < dw.size(); ++j) {
                    dw[j] *= rcp_batch_size;
                }

                opt->update(dw, w);
            }
        }
        clear_grads();
        post_update();
    }

    void layer::set_sample_count(size_t sample_count) {
        auto resize = [sample_count](tensor_t* tensor) {
            tensor->resize(sample_count, (*tensor)[0]);
        };

        for (size_t i = 0; i < in_concept_; ++i) {
            if (!is_trainable_concept(in_type_[i])) {
                resize(ith_in_node(i)->get_data());
            }
            resize(ith_in_node(i)->get_gradient());
        }

        for (size_t i = 0; i < out_concept_; ++i) {
            if (!is_trainable_concept(out_type_[i])) {
                resize(ith_out_node(i)->get_data());
            }
            resize(ith_out_node(i)->get_gradient());
        }
    }

    void layer::alloc_input(size_t i) const {
        prev_[i] = std::make_shared<edge>(nullptr, in_shape()[i], in_type_[i]);
    }

    void layer::alloc_output(size_t i) const {
        next_[i] = std::make_shared<edge>((layer*) this, out_shape()[i], out_type_[i]);
    }

    edgeptr_t layer::ith_in_node(size_t i) {
        if (!prev_[i]) alloc_input(i);
        return prev_[i];
    }

    edgeptr_t layer::ith_out_node(size_t i) {
        if (!next_[i]) alloc_output(i);
        return next_[i];
    }

    mat_t *layer::get_weight_data(size_t i) {
        assert(is_trainable_concept(in_type_[i]));
        return &(*ith_in_node(i)->get_data())[0];
    }

    const mat_t *layer::get_weight_data(size_t i) const {
        assert(is_trainable_concept(in_type_[i]));
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

        if (out_shape.size() != in_shape.size()) {
            connection_mismatch(*last_node, *next_node);
        }

        if (!last_node->next_[last_node_data_concept_idx]) {
            xs_error("Not alloc \"data\" on last node");
        }

        next_node->prev_[last_node_data_concept_idx] = last_node->next_[next_node_data_concept_idx];
    }

    // TODO: расширить описание
    void connection_mismatch(const layer& from, const layer& to) {
        throw xs_error("Connection mismatch error");
    }


} // xsdnn