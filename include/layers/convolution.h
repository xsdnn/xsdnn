//
// Created by rozhin on 21.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_CONVOLUTION_H
#define XSDNN_CONVOLUTION_H

#include "layer.h"
#include "../utils/util.h"
#include "../core/framework/params.h"

namespace xsdnn {

class conv : public layer {
public:
    explicit conv (size_t in_channel,
                   size_t in_height,
                   size_t in_width,
                   size_t out_channel,
                   size_t group_count = 1,
                   bool has_bias = true,
                   std::vector<size_t> kernel_shape = {},
                   std::vector<size_t> stride_shape = {},
                   std::vector<size_t> dilation_shape = {},
                   padding_mode pad_type = padding_mode::valid,
                   std::vector<size_t> pads = {})
        : conv(shape3d(in_channel, in_height, in_width), out_channel,
               group_count, has_bias, kernel_shape, stride_shape,
               dilation_shape, pad_type, pads) {}

    explicit conv (shape3d in_shape,
                   size_t out_channel,
                   size_t group_count = 1,
                   bool has_bias = true,
                   std::vector<size_t> kernel_shape = {},
                   std::vector<size_t> stride_shape = {},
                   std::vector<size_t> dilation_shape = {},
                   padding_mode pad_type = padding_mode::valid,
                   std::vector<size_t> pads = {})
        : layer({define_input_bias_condition(has_bias)}, {tensor_type::data}) {
        set_params(in_shape.C, in_shape.H, in_shape.W, out_channel,
                   group_count, has_bias, kernel_shape, stride_shape, dilation_shape, pad_type, pads);
    }

public:
    std::vector<shape3d> in_shape() const;
    std::vector<shape3d> out_shape() const;
    std::string layer_type() const;
    size_t fan_in_size() const {}
    size_t fan_out_size() const {}

    void
    forward_propagation(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>& out_data);

    void
    back_propagation(const std::vector<tensor_t*>& in_data,
                     const std::vector<tensor_t*>& out_data,
                     std::vector<tensor_t*>&       out_grad,
                     std::vector<tensor_t*>&       in_grad);

private:
    void set_params(size_t in_channel, size_t in_height, size_t in_width,
                    size_t out_channel, size_t group_count, bool has_bias,
                    std::vector<size_t> kernel_shape,
                    std::vector<size_t> stride_shape,
                    std::vector<size_t> dilation_shape,
                    padding_mode pad_type,
                    std::vector<size_t> pads);

private:
    params::conv params_;
};

} // xsdnn

#endif //XSDNN_CONVOLUTION_H
