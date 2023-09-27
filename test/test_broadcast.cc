//
// Created by rozhin on 06.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <gtest/gtest.h>
#include "../include/xsdnn.h"
#include "../include/utils/broadcaster.h"
#include "test_utils.h"
using namespace xsdnn;

TEST(broadcast, simple_1D_input0scalar) {
    shape3d shape1(1, 1, 1);
    shape3d shape2(1, 1, 3);

    std::vector<float> input0 = {2};
    std::vector<float> input1 = {1, 2, 3};
    std::vector<float> output = {0, 0, 0};

    tensor_t TensorInput0(XsDtype::F32, shape3d(1, 1, 1), nullptr);
    utils::vector_init(TensorInput0.GetMutableData<float>(), input0);

    tensor_t TensorInput1(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(TensorInput1.GetMutableData<float>(), input1);

    tensor_t TensorOutput(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(TensorOutput.GetMutableData<float>(), output);

    input_broadcaster in_bc(shape1, TensorInput0, &shape2, &TensorInput1);
    output_broadcaster out_bc(in_bc.get_span_size(), TensorOutput, in_bc.get_output_shape());

    broadcast bc(in_bc, out_bc);

    BroadcastFuncHolder func {
            [](broadcast& bc) {
                gsl::span<float> Output = bc.GetOutputSpan<float>();
                gsl::span<const float> Input = bc.GetSpanInput1<float>();
                const float Value = bc.GetScalarInput0<float>();
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
    gsl::span<const float> TensorOutputSpan = TensorOutput.GetDataAsSpan<float>();
    ASSERT_EQ(TensorOutputSpan[0], 3);
    ASSERT_EQ(TensorOutputSpan[1], 4);
    ASSERT_EQ(TensorOutputSpan[2], 5);
}

TEST(broadcast, simple_1D_input1scalar) {
    shape3d shape1(1, 1, 3);
    shape3d shape2(1, 1, 1);

    std::vector<float> input0 = {1, 2, 3};
    std::vector<float> input1 = {2};
    std::vector<float> output = {0, 0, 0};

    tensor_t TensorInput0(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(TensorInput0.GetMutableData<float>(), input0);

    tensor_t TensorInput1(XsDtype::F32, shape3d(1, 1, 1), nullptr);
    utils::vector_init(TensorInput1.GetMutableData<float>(), input1);

    tensor_t TensorOutput(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(TensorOutput.GetMutableData<float>(), output);

    input_broadcaster in_bc(shape1, TensorInput0, &shape2, &TensorInput1);
    output_broadcaster out_bc(in_bc.get_span_size(), TensorOutput, in_bc.get_output_shape());

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
    gsl::span<const float> TensorOutputSpan = TensorOutput.GetDataAsSpan<float>();
    ASSERT_EQ(TensorOutputSpan[0], 3);
    ASSERT_EQ(TensorOutputSpan[1], 4);
    ASSERT_EQ(TensorOutputSpan[2], 5);
}

TEST(broadcast, simple3D_1D) {
    shape3d shape1(3, 64, 64);
    shape3d shape2(3, 1, 1);

    std::vector<float> input0(shape1.size(), 1);
    std::vector<float> input1(shape2.size(), 2);
    std::vector<float> output(shape1.size(), 0);

    tensor_t TensorInput0(XsDtype::F32, shape1, nullptr);
    utils::vector_init(TensorInput0.GetMutableData<float>(), input0);

    tensor_t TensorInput1(XsDtype::F32, shape2, nullptr);
    utils::vector_init(TensorInput1.GetMutableData<float>(), input1);

    tensor_t TensorOutput(XsDtype::F32, shape1, nullptr);
    utils::vector_init(TensorOutput.GetMutableData<float>(), output);

    input_broadcaster in_bc(shape1, TensorInput0, &shape2, &TensorInput1);
    output_broadcaster out_bc(in_bc.get_span_size(), TensorOutput, in_bc.get_output_shape());

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
    gsl::span<const float> OutSpan = TensorOutput.GetDataAsSpan<float>();
    for (size_t i = 0; i < OutSpan.size(); ++i) {
        ASSERT_EQ(OutSpan[i], 2);
    }
}

TEST(broadcast, general) {
    shape3d shape1(1, 4, 3);
    shape3d shape2(1, 1, 3);

    std::vector<float> input0 = {
            0,  0,  0,
            10, 10, 10,
            20, 20, 20,
            30, 30, 30
    };
    std::vector<float> input1 = {1, 2, 3};
    std::vector<float> output(shape1.area(), 0);

    std::vector<float> expected = {
            1,  2,  3,
            11, 12, 13,
            21, 22, 23,
            31, 32, 33
    };

    tensor_t TensorInput0(XsDtype::F32, shape3d(1, 4, 3), nullptr);
    utils::vector_init(TensorInput0.GetMutableData<float>(), input0);

    tensor_t TensorInput1(XsDtype::F32, shape3d(1, 1, 3), nullptr);
    utils::vector_init(TensorInput1.GetMutableData<float>(), input1);

    tensor_t TensorOutput(XsDtype::F32, shape3d(1, 4, 3), nullptr);
    utils::vector_init(TensorOutput.GetMutableData<float>(), output);

    tensor_t TensorExpected(XsDtype::F32, shape3d(1, 4, 3), nullptr);
    utils::vector_init(TensorExpected.GetMutableData<float>(), expected);

    input_broadcaster in_bc(shape1, TensorInput0, &shape2, &TensorInput1);
    output_broadcaster out_bc(in_bc.get_span_size(), TensorOutput, in_bc.get_output_shape());

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
    utils::ContainerEqual(TensorOutput, TensorExpected);
}




