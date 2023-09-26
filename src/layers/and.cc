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

    bool and_layer::contains_only_one_zero(const BTensor &mats) {
        // TODO: этот слой должен принимать только булевый тип
    }

    void and_layer::forward_propagation(const std::vector<BTensor*> &in_data,
                                   std::vector<BTensor*> &out_data) {
//        const tensor_t& in1 = *in_data[0];
//        const tensor_t& in2 = *in_data[1];
//        tensor_t& out = *out_data[0];
//
//#ifndef NDEBUG
//        if (!contains_only_one_zero(in1) || !contains_only_one_zero(in2)) {
//            throw xs_error("[And Layer] Input Data Must be only 0/1 tensor.");
//        }
//#endif
//
//        concurrency::TryParallelFor(this->parallelize_, this->num_threads_, in1.size(), [&](size_t sample) {
//            const mat_t& in1_sample = in1[sample];
//            const mat_t& in2_sample = in2[sample];
//            mat_t& out_sample = out[sample];
//
//            mmpack::MmMulAdd(in1_sample.data(), in2_sample.data(), out_sample.data(), in1_sample.size());
//        });
    }

} // xsdnn
