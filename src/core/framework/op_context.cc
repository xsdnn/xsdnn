//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/framework/op_context.h>

namespace xsdnn {
    namespace core {

BTensor &OpContext::input_data(const size_t index) {
    return *(*in_data_)[index];
}

const BTensor &OpContext::input_data(const size_t index) const {
    return *(*in_data_)[index];
}

BTensor &OpContext::output_data(size_t index) {
    return *(*out_data_)[index];
}

const BTensor &OpContext::output_data(const size_t index) const {
    return *(*out_data_)[index];
}

BTensor &OpContext::input_grad(const size_t index) {
    return *(*in_grad_)[index];
}

const BTensor &OpContext::input_grad(const size_t index) const {
    return *(*in_grad_)[index];
}

BTensor &OpContext::output_grad(const size_t index) {
    return *(*out_grad_)[index];
}

const BTensor &OpContext::output_grad(const size_t index) const {
    return *(*out_grad_)[index];
}

void OpContext::set_in_out(const std::vector<BTensor *>& in_data, std::vector<BTensor *> &out_data) {
    in_data_ = const_cast<std::vector<BTensor*>*>(&in_data);
    out_data_ = &out_data;
}

void OpContext::set_in_out(const std::vector<BTensor *> &in_data, const std::vector<BTensor *> &out_data,
                           std::vector<BTensor *> &out_grad, std::vector<BTensor *> &in_grad) {
    in_data_ = const_cast<std::vector<BTensor*>*>(&in_data);
    out_data_ = const_cast<std::vector<BTensor*>*>(&out_data);
    in_grad_ = &in_grad;
    out_grad_ = &out_grad;
}

void OpContext::set_engine(core::backend_t engine) {
    engine_ = engine;
}

core::backend_t OpContext::engine() const {
    return engine_;
}

void OpContext::set_parallelize(bool parallelize) {
    parallelize_ = parallelize;
}

void OpContext::set_num_threads(size_t num_threads) {
    num_threads_ = num_threads;
}

bool OpContext::parallelize() const {
    return parallelize_;
}

size_t OpContext::num_threads() const {
    return num_threads_;
}

    } // core
} // xsdnn