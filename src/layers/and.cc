//
// Created by rozhin on 27.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/and.h>
#include <utils/macro.h>
#include <core/framework/threading.h>

namespace xsdnn {

    void and_layer::set_in_shape(const xsdnn::shape3d in_shape) {
        shape_ = in_shape;
    }

    std::vector<shape3d> and_layer::in_shape() const {
        return {shape_, shape_};
    }

    std::vector<shape3d> and_layer::out_shape() const {
        return {shape_};
    }

    std::string and_layer::layer_type() const {
        return "and_layer";
    }

    bool and_layer::contains_only_one_zero(const tensor_t &mats) {
        for (auto& mat : mats) {
            if (!std::all_of(mat.begin(), mat.end(), [](mm_scalar value) {
                return value == mm_scalar(0.0f) || value == mm_scalar(1.0f);
            })) {
                return false;
            }
        } return true;
    }

    void and_layer::forward_propagation(const std::vector<tensor_t *> &in_data,
                                   std::vector<tensor_t *> &out_data) {
        const tensor_t& in1 = *in_data[0];
        const tensor_t& in2 = *in_data[1];
        tensor_t& out = *out_data[0];

#ifndef NDEBUG
        if (!contains_only_one_zero(in1) || !contains_only_one_zero(in2)) {
            throw xs_error("[And Layer] Input Data Must be only 0/1 tensor.");
        }
#endif

        concurrency::TryParallelFor(this->parallelize_, this->num_threads_, in1.size(), [&](size_t sample) {
            const mat_t& in1_sample = in1[sample];
            const mat_t& in2_sample = in2[sample];
            mat_t& out_sample = out[sample];

            // TODO: impl simd
            throw xs_error("NotImplementedYet");
        });
    }

    void and_layer::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                                std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
        XS_UNUSED_PARAMETER(out_data);
        const tensor_t& in1 = *in_data[0];
        const tensor_t& in2 = *in_data[1];

        tensor_t& d_in1 = *in_grad[0];
        tensor_t& d_in2 = *in_grad[1];
        const tensor_t& dLz = *out_grad[0];

        concurrency::TryParallelFor(this->parallelize_, this->num_threads_, in1.size(), [&](size_t sample) {
            const mat_t& in1_sample = in1[sample];
            const mat_t& in2_sample = in2[sample];
            mat_t& d_in1_sample = d_in1[sample];
            mat_t& d_in2_sample = d_in2[sample];
            const mat_t& dLz_sample = dLz[sample];

            // TODO: impl simd
            throw xs_error("NotImplementedYet");
        });
    }

} // xsdnn
