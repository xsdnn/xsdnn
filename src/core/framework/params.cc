//
// Created by rozhin on 23.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/framework/params.h>

namespace xsdnn {
    namespace params {

conv::conv() {}

void
conv::infer_output_requirement_shape(xsdnn::shape3d in, size_t out_channel, size_t group_count,
                                     bool has_bias, std::vector<size_t> kernel_shape,
                                     std::vector<size_t> stride_shape, std::vector<size_t> dilation_shape,
                                     xsdnn::padding_mode pad_type, std::vector<size_t> pads,
                                     MmActivationType activation_type) {
    if (!is_init()) {
        throw xs_error("conv param doesn't init");
    }

    _.GroupCount = group_count;
    _.Bias = has_bias;
    pad_type_ = pad_type;
    activation_type_ = activation_type;

    if (_.Dimensions == 2) {
        this->_2D(in, out_channel, kernel_shape, stride_shape, dilation_shape, pad_type, pads);
    } else if (_.Dimensions == 1) {
        throw xs_error("conv param for 1D not implemented yet");
    } else {
        throw xs_error("[conv] unsupported dimensions in input data");
    }
}

bool conv::is_init() {
    return _.Dimensions == 1 || _.Dimensions == 2;
}

void conv::_2D(xsdnn::shape3d in, size_t out_channel, std::vector<size_t> kernel_shape,
               std::vector<size_t> stride_shape, std::vector<size_t> dilation_shape,
               xsdnn::padding_mode pad_type, std::vector<size_t> pads) {
    size_t rank = _.Dimensions;
    if (kernel_shape.size() != rank) throw xs_error("[conv] kernel_shape rank must be equal 2"); // TODO: здесь должна быть более полная проверка
    if (stride_shape.empty()) {
        stride_shape.resize(rank, 1);
    } else if (stride_shape.size() != rank){
        throw xs_error("[conv] stride_shape rank must be equal 2");
    }
    if (dilation_shape.empty()) {
        dilation_shape.resize(rank, 1);
    } else if (dilation_shape.size() != rank){
        throw xs_error("[conv] dilation_shape rank must be equal 2");
    }

    if (pads.empty()) {
        pads.resize(rank * 2, 0);
    } else if (pads.size() != 2 * rank) {
        throw xs_error("[conv] pads_shape rank must be equal 4");
    }

    assert(in.C % _.GroupCount == 0);
    assert(out_channel % _.GroupCount == 0);

    _.InChannel = static_cast<size_t>(in.C / _.GroupCount);
    _.InShape[0] = in.H;
    _.InShape[1] = in.W;


    size_t in_size = 1;
    size_t out_size = 1;
    size_t k = _.InChannel;

    for (size_t i = 0; i < rank; ++i) {
        computePad(pad_type, pads[i], pads[i + rank]);
        _.Padding[i] = pads[i];
        _.Padding[i + rank] = pads[i + rank];

        _.OutShape[i] = computeOutShape(_.InShape[i], kernel_shape[i],
                                            stride_shape[i], dilation_shape[i], pads[i], pads[i + rank]);

        _.KernelShape[i] = kernel_shape[i];
        _.DilationShape[i] = dilation_shape[i];
        _.StrideShape[i] = stride_shape[i];

        in_size *= _.InShape[i];
        out_size *= _.OutShape[i];
        k *= _.KernelShape[i];
    }
    _.FilterCount = static_cast<size_t>(out_channel / _.GroupCount);

    _.InSize = in_size;
    _.OutSize = out_size;
    _.K = k;

    this->computeAlgorithm();
    this->computeTmpBufferSize();
}

size_t conv::computeOutShape(const size_t in_dim, size_t kernel, size_t stride, size_t dilation, size_t pad_0,
                             size_t pad_1) {
    const size_t dkernel = dilation * (kernel - 1) + 1;
    return static_cast<size_t>(static_cast<float>(in_dim + pad_0 + pad_1 - dkernel) / stride + 1);
}

void conv::computePad(const xsdnn::padding_mode pad_type, size_t &pad_0, size_t &pad_1) {
    switch(pad_type) {
        case padding_mode::notset:
            break;
        case padding_mode::valid:
            pad_0 = 0;
            pad_1 = 0;
            break;
        default:
            throw xs_error("[conv ComputePad] Unsupported type of padding");
    }
}

void conv::computeAlgorithm() {
    _.Algorithm = _.Im2ColThenGemm;
}

void conv::computeTmpBufferSize() {
    _.TemproraryBufferSize = 16384;
}

    } // params
} // xsdnn