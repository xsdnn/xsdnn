//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/grad_checker.h>

namespace xsdnn {
    namespace internal {

std::vector<tensor_t> generate_fwd_data(const size_t num_concept,
                                        const std::vector<size_t> sizes) {
    std::vector<tensor_t> data;
    data.resize(num_concept);
    for (size_t i = 0; i < num_concept; ++i) {
        data[i].resize(1);
        data[i][0].resize(sizes[i]);
        uniform_rand(&data[i][0][0], sizes[i], -1.0f, 1.0f);
    }
    return data;
}

std::vector<tensor_t*> tensor2ptr(std::vector<tensor_t> &input) {
    std::vector<tensor_t *> ret(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        ret[i] = &input[i];
    }
    return ret;
}

} // internal

GradChecker::GradChecker(layer *l_ptr, mode m = mode::random) : l_ptr_(l_ptr), mode_(m) {}
GradChecker::GradChecker(const GradChecker &) = default;
GradChecker& GradChecker::operator=(const xsdnn::GradChecker &) = default;

GradChecker::status GradChecker::run() {
    size_t in_concept = l_ptr_->in_concept();
    size_t out_concept = l_ptr_->out_concept();

    std::vector<size_t> concept_sizes;
    for (size_t i = 0; i < in_concept; ++i) {
        concept_sizes.push_back(l_ptr_->in_shape()[i].size());
    }

    std::vector<tensor_t> in_data = internal::generate_fwd_data(in_concept, concept_sizes);
    /*
     * Next we need allocate buffer's
     */
    std::vector<tensor_t> in_grad = in_data;

    std::vector<size_t> out_concept_sizes;
    for (size_t i = 0; i < out_concept; ++i) {
        out_concept_sizes.push_back(l_ptr_->out_shape()[i].size());
    }
    std::vector<tensor_t> out_data = internal::generate_fwd_data(out_concept, out_concept_sizes);
    std::vector<tensor_t> out_grad = internal::generate_fwd_data(out_concept, out_concept_sizes);

    if (mode_ == mode::random) {
        for (size_t i = 0; i < 1000; ++i) {
            size_t in_concept_idx = uniform_idx(in_concept);
            size_t out_concept_idx = uniform_idx(out_concept);
            size_t in_position_idx = uniform_idx(in_data[in_concept_idx][0].size());
            size_t out_position_idx = uniform_idx(out_data[out_concept_idx][0].size());

            mm_scalar ngrad = numeric_gradient(l_ptr_,
                                               in_data,
                                               out_data,
                                               in_concept_idx,
                                               out_concept_idx,
                                               in_position_idx,
                                               out_position_idx);
            mm_scalar agrad = analytical_gradient(l_ptr_,
                                                  in_data,
                                                  out_data,
                                                  in_grad,
                                                  out_grad,
                                                  in_concept_idx,
                                                  out_concept_idx,
                                                  in_position_idx,
                                                  out_position_idx);
#if defined(MM_USE_DOUBLE)
            if (std::abs(ngrad - agrad) >= 1e-4) return status::bad;
#else
            if (std::abs(ngrad - agrad) >= 1e-2f)  return status::bad;
#endif
        }
    } else if (mode_ == mode::full) {
        throw xs_error("Not implemented yet");
    }  else {
        throw xs_error("Unsupported mode");
    }
    return status::ok;
}

mm_scalar GradChecker::numeric_gradient(layer *l_ptr, std::vector<tensor_t> in_data,
                                   std::vector<tensor_t> out_data, const size_t in_concept_idx,
                                   const size_t out_concept_idx, const size_t in_position_idx,
                                   const size_t out_position_idx) {
    mm_scalar h = std::sqrt(std::numeric_limits<mm_scalar>::epsilon());
    std::vector<tensor_t*> in_ = internal::tensor2ptr(in_data);
    std::vector<tensor_t*> out_ = internal::tensor2ptr(out_data);
    for (auto &tensor : out_data) tensorize::fill(tensor, 0.0f);
    mm_scalar prev_value = (*in_[in_concept_idx])[0][in_position_idx];
    (*in_[in_concept_idx])[0][in_position_idx] = prev_value + h;
    l_ptr_->forward_propagation(in_, out_);
    mm_scalar out_1 = (*out_[out_concept_idx])[0][out_position_idx];
    (*in_[in_concept_idx])[0][in_position_idx] = prev_value - h;
    l_ptr_->forward_propagation(in_, out_);
    mm_scalar out_2 = (*out_[out_concept_idx])[0][out_position_idx];
    return (out_1 - out_2) / (2 * h);
}

mm_scalar GradChecker::analytical_gradient(layer *l_ptr, std::vector<tensor_t> in_data, std::vector<tensor_t> out_data,
                                     std::vector<tensor_t> in_grad, std::vector<tensor_t> out_grad,
                                     const size_t in_concept_idx, const size_t out_concept_idx,
                                     const size_t in_position_idx, const size_t out_position_idx) {
    std::vector<tensor_t*> in_data_  = internal::tensor2ptr(in_data);
    std::vector<tensor_t*> in_grads_ = internal::tensor2ptr(in_grad);
    std::vector<tensor_t*> out_data_ = internal::tensor2ptr(out_data);
    for (auto &tensor : in_grad) tensorize::fill(tensor, 0.0f);
    for (auto &tensor : out_grad) tensorize::fill(tensor, 0.0f);
    for (auto &tensor : out_data) tensorize::fill(tensor, 0.0f);
    std::vector<tensor_t *> out_grads_ = internal::tensor2ptr(out_grad);
    out_grad[out_concept_idx][0][out_position_idx]    = 1.0f;  // set target grad to 1.
    l_ptr->forward_propagation(in_data_, out_data_);
    l_ptr->back_propagation(in_data_, out_data_, out_grads_, in_grads_);
    return in_grad[in_concept_idx][0][in_position_idx];
}

} // xsdnn