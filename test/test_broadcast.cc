//
// Created by rozhin on 06.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <gtest/gtest.h>
#include "../include/xsdnn.h"
#include "../include/utils/broadcaster.h"
#include "test_utils.h"
#include <cstring>
using namespace xsdnn;

TEST(broadcast, simple_1D_input0scalar_fp32) {
    shape3d shape1(1, 1, 1);
    shape3d shape2(1, 1, 3);

    mat_t input0(1 * dtype2sizeof(kXsFloat32));
    mat_t input1(3 * dtype2sizeof(kXsFloat32));
    mat_t output(3 * dtype2sizeof(kXsFloat32));

    utils::initializer_list_init_fp32(input0, {2});
    utils::initializer_list_init_fp32(input1, {1, 2, 3});
    utils::initializer_list_init_fp32(output, {0, 0, 0});

    input_broadcaster in_bc(kXsFloat32, shape1, input0, &shape2, &input1);
    output_broadcaster out_bc(kXsFloat32, in_bc.get_span_size(), output, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
            [](broadcast& bc) {
                gsl::span<float> Output = bc.GetOutputSpan<float>();
                gsl::span<const float> Input = bc.GetSpanInput1<float>();
                const float Value = bc.GetScalarInput0<float>();
                std::memcpy(Output.data(), Input.data(), Output.size());
                MmAdd(Value, Output.data(), Output.size() / sizeof(float));
            },
            [](broadcast& bc) {
                std::cout << "Call input1scalar" << std::endl;
            },
            [](broadcast& bc) {
                std::cout << "Call general" << std::endl;
            },
    };

    BroadcastKernelLoop(bc, func);
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&output);
    ASSERT_EQ(OutSpan[0], 3);
    ASSERT_EQ(OutSpan[1], 4);
    ASSERT_EQ(OutSpan[2], 5);
}

TEST(broadcast, simple_1D_input1scalar_fp32) {
    shape3d shape1(1, 1, 3);
    shape3d shape2(1, 1, 1);

    mat_t input0(3 * dtype2sizeof(kXsFloat32));
    mat_t input1(1 * dtype2sizeof(kXsFloat32));
    mat_t output(3 * dtype2sizeof(kXsFloat32));

    utils::initializer_list_init_fp32(input0, {1, 2, 3});
    utils::initializer_list_init_fp32(input1, {2});
    utils::initializer_list_init_fp32(output, {0, 0, 0});

    input_broadcaster in_bc(kXsFloat32, shape1, input0, &shape2, &input1);
    output_broadcaster out_bc(kXsFloat32, in_bc.get_span_size(), output, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
        [](broadcast& bc) {
            std::cout << "Call input0scalar" << std::endl;
        },
        [](broadcast& bc) {
            gsl::span<float> Output = bc.GetOutputSpan<float>();
            gsl::span<const float> Input = bc.GetSpanInput0<float>();
            const float Value = bc.GetScalarInput1<float>();
            std::memcpy(Output.data(), Input.data(), Output.size());
            MmAdd(Value, Output.data(), Output.size() / sizeof(float));
        },
        [](broadcast& bc) {
            std::cout << "Call general" << std::endl;
        },
    };

    BroadcastKernelLoop(bc, func);
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&output);
    ASSERT_EQ(OutSpan[0], 3);
    ASSERT_EQ(OutSpan[1], 4);
    ASSERT_EQ(OutSpan[2], 5);
}

TEST(broadcast, simple3D_1D_fp32) {
    shape3d shape1(3, 64, 64);
    shape3d shape2(3, 1, 1);

    mat_t in1(shape1.size() * dtype2sizeof(kXsFloat32));
    mat_t in2(shape2.size() * dtype2sizeof(kXsFloat32));
    mat_t out(shape1.size() * dtype2sizeof(kXsFloat32));

    utils::value_init_fp32(in1, 1);
    utils::value_init_fp32(in2, 2);
    utils::value_init_fp32(out, 0);

    input_broadcaster in_bc(kXsFloat32, shape1, in1, &shape2, &in2);
    output_broadcaster out_bc(kXsFloat32, in_bc.get_span_size(), out, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
            [](broadcast& bc) {
                std::cout << "Call input0scalar" << std::endl;
            },
            [](broadcast& bc) {
                gsl::span<float> Output = bc.GetOutputSpan<float>();
                gsl::span<const float> Input = bc.GetSpanInput0<float>();
                const float Value = bc.GetScalarInput1<float>();

                for (size_t i = 0; i < Input.size() / sizeof(float); ++i) {
                    Output[i] = Input[i] * Value;
                }
            },
            [](broadcast& bc) {
                std::cout << "Call general" << std::endl;
            },
    };

    BroadcastKernelLoop(bc, func);
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], 2.0f, kXsFloat32);
    }
}

TEST(broadcast, general_fp32) {
    shape3d shape1(1, 4, 3);
    shape3d shape2(1, 1, 3);

    mat_t in1(shape1.size() * dtype2sizeof(kXsFloat32));
    mat_t in2(shape2.size() * dtype2sizeof(kXsFloat32));
    mat_t out(shape1.area() * dtype2sizeof(kXsFloat32));

    utils::initializer_list_init_fp32(in1, {
            0,  0,  0,
            10, 10, 10,
            20, 20, 20,
            30, 30, 30
    });
    utils::initializer_list_init_fp32(in2, {1, 2, 3});
    utils::value_init_fp32(out, 0);

    mat_t expected(shape1.area() * dtype2sizeof(kXsFloat32));
    utils::initializer_list_init_fp32(expected, {
            1,  2,  3,
            11, 12, 13,
            21, 22, 23,
            31, 32, 33
    });

    input_broadcaster in_bc(kXsFloat32, shape1, in1, &shape2, &in2);
    output_broadcaster out_bc(kXsFloat32, in_bc.get_span_size(), out, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
            [](broadcast& bc) {
                std::cout << "Call input0scalar" << std::endl;
            },
            [](broadcast& bc) {
                std::cout << "Call input1scalar" << std::endl;
            },
            [](broadcast& bc) {
                gsl::span<float> Output =  bc.GetOutputSpan<float>();
                gsl::span<const float> In0 = bc.GetSpanInput0<float>();
                gsl::span<const float> In1 = bc.GetSpanInput1<float>();
                for (size_t i = 0; i < Output.size() / sizeof(float); ++i) {
                    Output[i] = In0[i] + In1[i];
                }
            },
    };

    BroadcastKernelLoop(bc, func);
    gsl::span<const float> OutSpan = GetDataAsSpan<const float>(&out);
    gsl::span<const float> ExpectedSpan = GetDataAsSpan<const float>(&expected);

    for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
        utils::xsAssert_eq(OutSpan[i], ExpectedSpan[i], kXsFloat32);
    }
}




