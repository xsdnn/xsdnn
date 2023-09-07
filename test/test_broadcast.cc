//
// Created by rozhin on 06.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <gtest/gtest.h>
#include "../include/xsdnn.h"
#include "../include/utils/broadcaster.h"
using namespace xsdnn;

TEST(broadcast, simple_1D_input0scalar) {
    shape3d shape1(1, 1, 1);
    shape3d shape2(1, 1, 3);

    mat_t input0 = {2};
    mat_t input1 = {1, 2, 3};
    mat_t output = {0, 0, 0};

    input_broadcaster in_bc(shape1, input0, &shape2, &input1);
    output_broadcaster out_bc(in_bc.get_span_size(), output, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
            [](broadcast& bc) {
                gsl::span<mm_scalar> Output = bc.GetOutputSpan<mm_scalar>();
                gsl::span<const mm_scalar> Input = bc.GetSpanInput1<mm_scalar>();
                const mm_scalar Value = bc.GetScalarInput0<mm_scalar>();
                std::copy(Input.begin(), Input.end(), Output.begin());
                MmAdd(Value, Output.data(), Output.size());
            },
            [](broadcast& bc) {
                std::cout << "Call input1scalar" << std::endl;
            },
            [](broadcast& bc) {
                std::cout << "Call general" << std::endl;
            },
    };

    BroadcastKernelLoop(bc, func);
    ASSERT_EQ(output[0], 3);
    ASSERT_EQ(output[1], 4);
    ASSERT_EQ(output[2], 5);
}

TEST(broadcast, simple_1D_input1scalar) {
    shape3d shape1(1, 1, 3);
    shape3d shape2(1, 1, 1);

    mat_t input0 = {1, 2, 3};
    mat_t input1 = {2};
    mat_t output = {0, 0, 0};

    input_broadcaster in_bc(shape1, input0, &shape2, &input1);
    output_broadcaster out_bc(in_bc.get_span_size(), output, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
        [](broadcast& bc) {
            std::cout << "Call input0scalar" << std::endl;
        },
        [](broadcast& bc) {
            gsl::span<mm_scalar> Output = bc.GetOutputSpan<mm_scalar>();
            gsl::span<const mm_scalar> Input = bc.GetSpanInput0<mm_scalar>();
            std::copy(Input.begin(), Input.end(), Output.begin());
            const mm_scalar Value = bc.GetScalarInput1<mm_scalar>();
            MmAdd(Value, Output.data(), Output.size());
        },
        [](broadcast& bc) {
            std::cout << "Call general" << std::endl;
        },
    };

    BroadcastKernelLoop(bc, func);

    ASSERT_EQ(output[0], 3);
    ASSERT_EQ(output[1], 4);
    ASSERT_EQ(output[2], 5);
}

TEST(broadcast, simple3D_1D) {
    shape3d shape1(3, 64, 64);
    shape3d shape2(3, 1, 1);

    mat_t in1(shape1.size(), 1);
    mat_t in2(shape2.size(), 2);
    mat_t out(shape1.size(), 0);

    input_broadcaster in_bc(shape1, in1, &shape2, &in2);
    output_broadcaster out_bc(in_bc.get_span_size(), out, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
            [](broadcast& bc) {
                std::cout << "Call input0scalar" << std::endl;
            },
            [](broadcast& bc) {
                gsl::span<mm_scalar> Output = bc.GetOutputSpan<mm_scalar>();
                gsl::span<const mm_scalar> Input = bc.GetSpanInput0<mm_scalar>();
                const mm_scalar Value = bc.GetScalarInput1<mm_scalar>();

                for (size_t i = 0; i < Input.size(); ++i) {
                    Output[i] = Input[i] * Value;
                }
            },
            [](broadcast& bc) {
                std::cout << "Call general" << std::endl;
            },
    };

    BroadcastKernelLoop(bc, func);
    for (size_t i = 0; i < out.size(); ++i) {
        ASSERT_EQ(out[i], 2);
    }
}

TEST(broadcast, general) {
    shape3d shape1(1, 4, 3);
    shape3d shape2(1, 1, 3);

    mat_t in1 = {
            0,  0,  0,
            10, 10, 10,
            20, 20, 20,
            30, 30, 30
    };
    mat_t in2 = {1, 2, 3};
    mat_t out(shape1.area(), 0);

    mat_t expected = {
            1,  2,  3,
            11, 12, 13,
            21, 22, 23,
            31, 32, 33
    };

    input_broadcaster in_bc(shape1, in1, &shape2, &in2);
    output_broadcaster out_bc(in_bc.get_span_size(), out, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
            [](broadcast& bc) {
                std::cout << "Call input0scalar" << std::endl;
            },
            [](broadcast& bc) {
                std::cout << "Call input1scalar" << std::endl;
            },
            [](broadcast& bc) {
                gsl::span<mm_scalar> Output =  bc.GetOutputSpan<mm_scalar>();
                gsl::span<const mm_scalar> In0 = bc.GetSpanInput0<mm_scalar>();
                gsl::span<const mm_scalar> In1 = bc.GetSpanInput1<mm_scalar>();
                for (size_t i = 0; i < Output.size(); ++i) {
                    Output[i] = In0[i] + In1[i];
                }
            },
    };

    BroadcastKernelLoop(bc, func);
    for (size_t i = 0; i < out.size(); ++i) {
        ASSERT_EQ(out[i], expected[i]);
    }
}




