//
// Created by rozhin on 15.11.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../include/xsdnn.h"
#include "test_utils.h"

void ProfilingConvXNNBackendEngine(size_t InputChannels,
                                   size_t InputHeight,
                                   size_t InputWidth,
                                   size_t OutChannel,
                                   size_t KernelH,
                                   size_t KernelW,
                                   size_t GroupCount,
                                   size_t Bias,
                                   size_t StrideH,
                                   size_t StrideW,
                                   size_t DilationH,
                                   size_t DilationW,
                                   size_t PadHTop,
                                   size_t PadWLeft,
                                   size_t PadHBottom,
                                   size_t PadWRight,
                                   MmActivationType act_type,
                                   core::backend_t eng,
                                   size_t NumRepeat) {
    size_t in_channel = InputChannels;
    size_t in_height = InputHeight;
    size_t in_width = InputWidth;
    size_t out_channel = OutChannel;
    size_t kernel_h = KernelH;
    size_t kernel_w = KernelW;
    size_t group_count = GroupCount;
    bool has_bias = Bias;
    size_t stride_h = StrideH;
    size_t stride_w = StrideW;
    size_t dilation_h = DilationH;
    size_t dilation_w = DilationW;
    size_t pads_h_top = PadHTop;
    size_t pads_w_left = PadWLeft;
    size_t pads_h_bottom = PadHBottom;
    size_t pads_w_right = PadWRight;
    MmActivationType activation_type = act_type;
    xsdnn::core::backend_t engine = eng;

    xsdnn::shape3d in_shape(in_channel, in_height, in_width);
    xsdnn::mat_t input(in_shape.size() * sizeof(float));
    utils::random_init_fp32(input);

    xsdnn::conv ConvOp(/*in_shape=*/in_shape, /*out_channel=*/out_channel, /*kernel_shape=*/{kernel_h, kernel_w},
            /*group_count=*/group_count, /*has_bias=*/has_bias, /*stride_shape=*/{stride_h, stride_w},
            /*dilation_shape=*/{dilation_h, dilation_w}, /*pad_type=*/xsdnn::padding_mode::notset,
            /*pads=*/{pads_h_top, pads_w_left, pads_h_bottom, pads_w_right},
            /*activation_type=*/activation_type, /*engine=*/engine);

    ConvOp.set_parallelize(false);
    ConvOp.set_num_threads(12);
    ConvOp.setup(true);
    ConvOp.set_in_data({{ input }});

    while (NumRepeat-- > 0)
        ConvOp.forward();
}

int main() {
    EASY_PROFILER_ENABLE
    xnn_status status = xnn_initialize(nullptr);
    /*************************** Conv 0 **************************/
    /*                            C    H    W    Cout    Kh    Kw     Gc     Bias     Sh    Sw    Dh   Dw   PHtop  PWleft   PHbottom    PWright                Activation                    Engine   N*/
    ProfilingConvXNNBackendEngine(3, 300, 300,     24,    3,    3,     1,       1,     2,    2,    1,   1,      0,      0,         1,         1, MmActivationType::NotSet, core::backend_t::xnnpack, 1000);
    profiler::dumpBlocksToFile("test_profile.prof");
}