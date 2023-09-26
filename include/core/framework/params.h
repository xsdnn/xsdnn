//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_PARAMS_H
#define XSDNN_PARAMS_H

#include <cstddef>
#include <unordered_map>
#include "tensor.h"
#include "../../mmpack/mmpack.h"
#include "../../utils/tensor_shape.h"
#include "../../utils/util.h"

namespace xsdnn {
    namespace params {

struct fully {
    size_t in_size_;
    size_t out_size_;
    bool   has_bias_;
};

struct bnorm {
    shape3d in_shape_;
    float momentum_;
    float eps_;
    op_mode phase_;
    std::unordered_map<std::string, tensor_t> stat_holder;

    bool statistic_initialized {false};
};

struct max_pool {
    shape3d in_shape_;
    shape3d out_shape_;
    size_t kernel_x_;
    size_t kernel_y_;
    size_t stride_x_;
    size_t stride_y_;
    padding_mode pad_type_;

    std::vector<std::vector<size_t>> out2in;
    std::vector<size_t> in2out;
};

struct global_avg_pool {
    shape3d in_shape_;
    shape3d out_shape_;
};

struct conv {
public:
    conv();

    void infer_output_requirement_shape(shape3d in, size_t out_channel, size_t group_count, bool has_bias,
                                        std::vector<size_t> kernel_shape,
                                        std::vector<size_t> stride_shape,
                                        std::vector<size_t> dilation_shape,
                                        padding_mode pad_type,
                                        std::vector<size_t> pads,
                                        mmpack::MmActivationType activation_type);

private:
    bool is_init();

    void _2D(shape3d in, size_t out_channel,
             std::vector<size_t> kernel_shape,
             std::vector<size_t> stride_shape,
             std::vector<size_t> dilation_shape,
             padding_mode pad_type,
             std::vector<size_t> pads);

    size_t computeOutShape(const size_t in_dim, size_t kernel, size_t stride, size_t dilation, size_t pad_0, size_t pad_1);
    void computePad(const padding_mode pad_type, size_t& pad_0, size_t& pad_1);
    void computeTmpBufferSize();
    void computeAlgorithm();

public:
    mmpack::MM_CONV_PARAMS _;
    padding_mode pad_type_;
    mmpack::MmActivationType activation_type_;
};

    } // params
} // xsdnn

#endif //XSDNN_PARAMS_H
