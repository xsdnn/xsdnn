//
// Created by rozhin on 21.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/convolution.h>

namespace xsdnn {

void conv::set_params(size_t in_channel, size_t in_height, size_t in_width, size_t out_channel, size_t group_count, bool has_bias,
                      std::vector<size_t> kernel_shape,
                      std::vector<size_t> stride_shape,
                      std::vector<size_t> dilation_shape,
                      padding_mode pad_type,
                      std::vector<size_t> pads) {
    if (is_1D_tensor(shape3d(in_channel, in_height, in_width))) {
        params_.holder_.dimensions_ = 1;
    } else if (is_2D_tensor(shape3d(in_channel, in_height, in_width))) {
        params_.holder_.dimensions_ = 2;
    } else {
        xs_error("Unsupported dimensions in input data of conv layer");
    }
    params_.infer_output_requirement_shape(shape3d(in_channel, in_height, in_width),
                                           group_count, has_bias,
                                           out_channel, kernel_shape, stride_shape,
                                           dilation_shape, pad_type, pads);
}

std::vector<shape3d> conv::in_shape() const {
    if (params_.holder_.dimensions_ == 2) {
        if (params_.holder_.has_bias_) {
            return {
                    shape3d(params_.holder_.in_shape_[0], params_.holder_.in_shape_[1], params_.holder_.in_shape_[2]),

                    shape3d(params_.holder_.filter_count_ * params_.holder_.out_shape_[0],
                            params_.holder_.kernel_shape_[0], params_.holder_.kernel_shape_[1]),

                    shape3d(params_.holder_.out_shape_[0], 1, 1)
            };
        } else {
            return {
                    shape3d(params_.holder_.in_shape_[0], params_.holder_.in_shape_[1], params_.holder_.in_shape_[2]),

                    shape3d(params_.holder_.filter_count_ * params_.holder_.out_shape_[0],
                            params_.holder_.kernel_shape_[0], params_.holder_.kernel_shape_[1])
            };
        }
    } else {
        throw xs_error("conv in_shape for 1D not implemented yet");
    }

}

std::vector<shape3d> conv::out_shape() const {
    if (params_.holder_.dimensions_ == 2) {
        return { shape3d(params_.holder_.out_shape_[0], params_.holder_.out_shape_[1], params_.holder_.out_shape_[2]) };
    } else {
        throw xs_error("conv out_shape for 1D not implemented yet");
    }
}

std::string conv::layer_type() const {
    return "conv";
}

void conv::forward_propagation(const std::vector<tensor_t *> &in_data,
                                   std::vector<tensor_t *> &out_data) {

}

void conv::back_propagation(const std::vector<tensor_t *> &in_data, const std::vector<tensor_t *> &out_data,
                            std::vector<tensor_t *> &out_grad, std::vector<tensor_t *> &in_grad) {

}

} // xsdnn
