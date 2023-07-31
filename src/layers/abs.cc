//
// Created by rozhin on 27.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/abs.h>
#include <utils/macro.h>
#include <core/framework/threading.h>

namespace xsdnn {

    void abs::set_in_shape(const xsdnn::shape3d in_shape) {
        shape_ = in_shape;
    }

    std::vector<shape3d> abs::in_shape() const {
        return {shape_};
    }

    std::vector<shape3d> abs::out_shape() const {
        return {shape_};
    }

    std::string abs::layer_type() const {
        return "abs";
    }

    void abs::forward_propagation(const std::vector<tensor_t *> &in_data,
                                  std::vector<tensor_t *> &out_data) {
        const tensor_t& in = *in_data[0];
        tensor_t& out = *out_data[0];

        concurrency::TryParallelFor(this->parallelize_, this->num_threads_, in.size(), [&](size_t sample) {
            for (size_t j = 0; j < in[sample].size(); ++j) {
                out[sample][j] = std::abs(in[sample][j]);
            }
        });
    }

    void abs::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                               std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {
        XS_UNUSED_PARAMETER(in_data);
        XS_UNUSED_PARAMETER(out_data);

        tensor_t& dx = *in_grad[0];
        const tensor_t& dLz = *out_grad[0];

        concurrency::TryParallelFor(this->parallelize_, this->num_threads_, dx.size(), [&](size_t sample) {
            for (size_t j = 0; j < dx[sample].size(); ++j) {
                dx[sample][j] = (mm_scalar(0.0f) < dLz[sample][j]) - (dLz[sample][j] < mm_scalar(0.0f));
            }
        });
    }

}