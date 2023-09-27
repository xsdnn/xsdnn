//
// Created by rozhin on 21.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include <gtest/gtest.h>
#include <sys/mman.h>
#include "test_utils.h"
using namespace xsdnn;

// TODO: проверить этот тест. In Shape для веса выдается некорректно
TEST(conv, _2D_params_check_1) {
    shape3d in(12, 255, 255);
    conv c(in, /*out_channel=*/ 6, /*kernel_shape=*/ {3, 3},
           /*group_count=*/ 3, /*has_bias=*/ true,
           /*stride_shape=*/ {1, 1}, /*dilation_shape=*/ {1, 1},
           /*pad_type=*/padding_mode::notset, /*pads=*/ {1, 1, 1, 1});

    params::conv P = c.get_params();

    ASSERT_EQ(P._.Dimensions, 2);
    ASSERT_EQ(P._.GroupCount, 3);
    ASSERT_EQ(P._.InChannel, 4);
    ASSERT_EQ(P._.InShape[0], 255);
    ASSERT_EQ(P._.InShape[1], 255);
    ASSERT_EQ(P._.InSize, 255 * 255);
    ASSERT_EQ(P._.OutShape[0], 255);
    ASSERT_EQ(P._.OutShape[1], 255);
    ASSERT_EQ(P._.OutSize, 255 * 255);
    ASSERT_EQ(P._.K, 4 * 3 * 3);
    ASSERT_EQ(P._.Padding[0], 1);
    ASSERT_EQ(P._.Padding[1], 1);
    ASSERT_EQ(P._.Padding[2], 1);
    ASSERT_EQ(P._.Padding[3], 1);
    ASSERT_EQ(P._.KernelShape[0], 3);
    ASSERT_EQ(P._.KernelShape[1], 3);
    ASSERT_EQ(P._.DilationShape[0], 1);
    ASSERT_EQ(P._.DilationShape[1], 1);
    ASSERT_EQ(P._.StrideShape[0], 1);
    ASSERT_EQ(P._.StrideShape[1], 1);
    ASSERT_EQ(P._.FilterCount, 2);
    ASSERT_EQ(P._.Algorithm, P._.Im2ColThenGemm);
    ASSERT_EQ(P._.Bias, true);
    ASSERT_EQ(P._.TemproraryBufferSize, 16384);
}

TEST(conv, simple_forward) {
    shape3d in_shape(3, 4, 4);
    conv c(in_shape, 3, {2, 2}, 3, true);

    const std::vector<float> Input = {
            -1.8776015043258667,
            -0.2049034833908081,
            -0.13616761565208435,
            0.8064142465591431,
            2.51012921333313,
            0.10977518558502197,
            -2.0887832641601562,
            -0.08098437637090683,
            -1.1682378053665161,
            -0.5134232044219971,
            0.9532246589660645,
            0.07082877308130264,
            -1.706060767173767,
            -0.1671934574842453,
            0.4446631669998169,
            2.1559534072875977,
            -0.27103009819984436,
            -1.2768625020980835,
            -1.788601279258728,
            -0.7554757595062256,
            -0.11542399227619171,
            -0.23354925215244293,
            1.6248456239700317,
            -1.7911633253097534,
            -1.0857563018798828,
            0.9385977983474731,
            1.0483120679855347,
            0.6492710113525391,
            -0.1565261036157608,
            -1.5730873346328735,
            -0.026409100741147995,
            1.0576292276382446,
            -0.00040333415381610394,
            -0.837020993232727,
            -0.5681514143943787,
            -1.3411329984664917,
            -0.3682830035686493,
            -0.3687058389186859,
            -0.6560876369476318,
            0.05378172919154167,
            -0.1029391959309578,
            -0.34406107664108276,
            -1.3667269945144653,
            -0.04167516157031059,
            -0.02375619299709797,
            -0.8656331896781921,
            -0.8146375417709351,
            -0.9757303595542908
    };
    const std::vector<float> Output = {
            0.8995431065559387,
            0.38253235816955566,
            0.2557677626609802,
            -0.8466297388076782,
            -0.28133895993232727,
            0.974159836769104,
            0.5159683227539062,
            0.41759151220321655,
            -0.42970991134643555,
            -0.01950083300471306,
            0.34620755910873413,
            -1.0064418315887451,
            0.38303855061531067,
            -0.1145615503191948,
            0.35485294461250305,
            -0.6674518585205078,
            0.29136741161346436,
            0.346919983625412,
            -0.0030910789500921965,
            0.1547432690858841,
            0.19265101850032806,
            -0.011673642322421074,
            0.17509226500988007,
            -0.5040093064308167,
            0.0075158062390983105,
            0.20843656361103058,
            0.011137520894408226
    };

    std::vector<float> Weights = {
        -0.4036754369735718,
        0.12995994091033936,
        0.0194014310836792,
        -0.08908277750015259,
        0.16426432132720947,
        -0.08304166793823242,
        -0.17363417148590088,
        0.2517896294593811,
        -0.2612036466598511,
        -0.3790665864944458,
        0.2879072427749634,
        -0.14387822151184082
    };

    std::vector<float> Bias = {
            0.1293, -0.0422, -0.2675
    };
    tensor_t TensorInput(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(TensorInput.GetMutableData<float>(), Input);
    tensor_t TensorExpected(XsDtype::F32, c.out_shape()[0], nullptr);
    utils::vector_init(TensorExpected.GetMutableData<float>(), Output);

    c.set_parallelize(false);
    c.setup(false);

    utils::vector_init(c.prev()[1]->get_data()->at(0).GetMutableData<float>(), Weights);
    utils::vector_init(c.prev()[2]->get_data()->at(0).GetMutableData<float>(), Bias);

    c.set_in_data({{ TensorInput }});
    c.forward();

    tensor_t OutTensor = c.output()[0][0];
    utils::ContainerNear(OutTensor, TensorExpected, 1e-3);
}

TEST(conv, simple_forward_without_bias) {
    shape3d in_shape(3, 4, 4);
    conv c(in_shape, 3, {2, 2}, 3, false);

    const std::vector<float> Input = {
            -0.8597232699394226,
            -0.016536127775907516,
            -1.067084789276123,
            -1.2771207094192505,
            0.5289744138717651,
            0.4591456353664398,
            -0.6136614680290222,
            -0.3110077679157257,
            -1.003212809562683,
            -0.25640687346458435,
            -0.34296318888664246,
            0.3661316931247711,
            1.96624755859375,
            0.12085486948490143,
            0.28439822793006897,
            -0.9817278385162354,
            1.723364233970642,
            1.11757230758667,
            -0.8552507758140564,
            -0.3802138864994049,
            0.791811466217041,
            0.450124055147171,
            0.1003960445523262,
            -0.6282744407653809,
            -0.10524702817201614,
            0.5840324759483337,
            -0.05916932225227356,
            1.0429607629776,
            -0.4981708824634552,
            1.1234641075134277,
            -0.5341346263885498,
            0.13873034715652466,
            -0.11888983845710754,
            -1.597569465637207,
            -1.7347184419631958,
            0.6859488487243652,
            0.27496105432510376,
            1.2040894031524658,
            0.4182891249656677,
            -0.04522772505879402,
            0.8157641887664795,
            -0.9951342940330505,
            -0.8859978914260864,
            0.6702648401260376,
            -0.1707228422164917,
            -1.0781022310256958,
            0.36826708912849426,
            -0.6138019561767578,
    };

    std::vector<float> Output = {
            -0.4404990077018738,
            0.12277835607528687,
            -0.21373416483402252,
            0.3472598195075989,
            0.21447788178920746,
            -0.31341293454170227,
            -0.44802138209342957,
            -0.22438563406467438,
            0.349630743265152,
            0.9372735619544983,
            0.01971816085278988,
            -0.26863399147987366,
            0.27968576550483704,
            0.28655096888542175,
            -0.17447705566883087,
            0.08845566213130951,
            0.39708858728408813,
            0.19920746982097626,
            0.044415123760700226,
            -0.07635539025068283,
            -0.6138955354690552,
            -0.44105473160743713,
            0.7539032697677612,
            0.1517699509859085,
            1.07393217086792,
            0.6412769556045532,
            -0.2617749273777008,
    };

    std::vector<float> Weights = {
            0.2444877028465271,
            0.09724676609039307,
            -0.0643761157989502,
            -0.42393046617507935,
            0.15800708532333374,
            0.3538004755973816,
            0.3129462003707886,
            0.0483817458152771,
            0.07881486415863037,
            -0.4501446485519409,
            -0.4511941075325012,
            -0.44954395294189453
    };

    tensor_t TensorInput(XsDtype::F32, in_shape, nullptr);
    utils::vector_init(TensorInput.GetMutableData<float>(), Input);

    tensor_t TensorExpected(XsDtype::F32, c.out_shape()[0], nullptr);
    utils::vector_init(TensorExpected.GetMutableData<float>(), Output);

    c.set_parallelize(false);
    c.setup(false);
    utils::vector_init(c.prev()[1]->get_data()->at(0).GetMutableData<float>(), Weights);
    c.set_in_data({{ TensorInput }});
    c.forward();

    tensor_t OutTensor = c.output()[0][0];
    utils::ContainerNear(OutTensor, TensorExpected, 1e-5);
}

TEST(conv, cerial) {
    shape3d in(12, 255, 255);
    conv c(in, /*out_channel=*/ 6, /*kernel_shape=*/ {3, 3},
            /*group_count=*/ 3, /*has_bias=*/ true,
            /*stride_shape=*/ {1, 1}, /*dilation_shape=*/ {1, 1},
            /*pad_type=*/padding_mode::notset, /*pads=*/ {1, 1, 1, 1});
    ASSERT_TRUE(utils::cerial_testing(c));
}

class SConvTester {
public:
    void ExecuteLong() {
        static const unsigned cs[] = {32, 14,};
        static const unsigned is[] = {53, 11, 5};

        for (unsigned i = 1; i <= 32; i++) {
            Test(18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
            Test(18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
            Test(18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        for (unsigned gc = 0; gc < _countof(cs); gc++) {
            for (unsigned ih = 0; ih < _countof(is); ih++) {
                for (unsigned iw = 0; iw < _countof(is); iw++) {
                    fprintf(stderr, "Handling depthwise %ux%ux%u\n", cs[gc], is[ih], is[iw]);
                    for (unsigned p0 = 0; p0 < 2; p0++) {
                        for (unsigned p1 = 0; p1 < 2; p1++) {
                            for (unsigned p2 = 0; p2 < 2; p2++) {
                                for (unsigned p3 = 0; p3 < 2; p3++) {
                                    for (unsigned dh = 1; dh <= 2; dh++) {
                                        for (unsigned dw = 1; dw <= 2; dw++) {
                                            for (unsigned sh = 1; sh <= 2; sh++) {
                                                for (unsigned sw = 1; sw <= 2; sw++) {
                                                    Test(cs[gc], 1, is[ih], is[iw], 1, 3, 3, p0, p1, p2, p3, dh, dw, sh, sw);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (unsigned ic = 0; ic < _countof(cs); ic++) {
            for (unsigned ih = 0; ih < _countof(is); ih++) {
                for (unsigned iw = 0; iw < _countof(is); iw++) {
                    fprintf(stderr, "Handling %ux%ux%u\n", cs[ic], is[ih], is[iw]);
                    for (unsigned fc = 0; fc < _countof(cs); fc++) {
                        for (unsigned kh = 1; kh <= 5; kh++) {
                            if (kh == 4) continue;
                            for (unsigned kw = 1; kw <= 5; kw++) {
                                if (kw == 4) continue;
                                for (unsigned p0 = 0; p0 < 2; p0++) {
                                    for (unsigned p1 = 0; p1 < 2; p1++) {
                                        for (unsigned p2 = 0; p2 < 2; p2++) {
                                            for (unsigned p3 = 0; p3 < 2; p3++) {
                                                for (unsigned dh = 1; dh <= 2; dh++) {
                                                    for (unsigned dw = 1; dw <= 2; dw++) {
                                                        for (unsigned sh = 1; sh <= 2; sh++) {
                                                            for (unsigned sw = 1; sw <= 2; sw++) {
                                                                Test(1, cs[ic], is[ih], is[iw], cs[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void Test(size_t GroupCount,
            size_t InputChannels,
            size_t InputHeight,
            size_t InputWidth,
            size_t FilterCount,
            size_t KernelHeight,
            size_t KernelWidth,
            size_t PaddingLeftHeight,
            size_t PaddingLeftWidth,
            size_t PaddingRightHeight,
            size_t PaddingRightWidth,
            size_t DilationHeight,
            size_t DilationWidth,
            size_t StrideHeight,
            size_t StrideWidth) {
        int64_t OutputHeight64 =
                ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
                 (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) /
                int64_t(StrideHeight) +
                1;
        int64_t OutputWidth64 =
                ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
                 (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) /
                int64_t(StrideWidth) +
                1;

        if (OutputHeight64 <= 0 || OutputWidth64 <= 0) {
            return;
        }

        size_t OutputHeight = size_t(OutputHeight64);
        size_t OutputWidth = size_t(OutputWidth64);

        size_t InputSize = InputHeight * InputWidth;
        size_t KernelSize = KernelHeight * KernelWidth;
        size_t OutputSize = OutputHeight * OutputWidth;

        size_t InputElements = GroupCount * InputChannels * InputSize;
        size_t FilterElements = GroupCount * FilterCount * InputChannels * KernelSize;
        size_t BiasElements = GroupCount * FilterCount;
        size_t OutputElements = GroupCount * FilterCount * OutputSize;

        const float* Input = BufferInput.GetBuffer(InputElements);
        const float* Filter = BufferFilter.GetBuffer(FilterElements);
        const float* Bias = BufferBias.GetBuffer(BiasElements);
        float* Output = BufferOutput.GetBuffer(OutputElements);
        float* OutputReference = BufferOutputReference.GetBuffer(OutputElements);

        MmConv2D(GroupCount,
                 InputChannels,
                 InputHeight, InputWidth,
                 FilterCount,
                 KernelHeight, KernelWidth,
                 PaddingLeftHeight, PaddingLeftWidth,
                 PaddingRightHeight, PaddingRightWidth,
                 DilationHeight, DilationWidth,
                 StrideHeight, StrideWidth,
                 OutputHeight, OutputWidth,
                 Input,
                 Filter,
                 Bias,
                 Output);

        ReferenceConv2D(GroupCount,
                        InputChannels,
                        InputHeight, InputWidth,
                        FilterCount,
                        KernelHeight, KernelWidth,
                        PaddingLeftHeight, PaddingLeftWidth,
                        DilationHeight, DilationWidth,
                        StrideHeight, StrideWidth,
                        OutputHeight, OutputWidth,
                        Input,
                        Filter,
                        Bias,
                        OutputReference);

        ASSERT_EQ(memcmp(Output, OutputReference, OutputElements * sizeof(float)), 0)
                                    << "G" << GroupCount << "/"
                                    << "Cpg" << InputChannels << "/"
                                    << "Fpg" << FilterCount << "/"
                                    << "H" << InputHeight << "/"
                                    << "W" << InputWidth << "/"
                                    << "KH" << KernelHeight << "/"
                                    << "KW" << KernelWidth << "/"
                                    << "Pad" << PaddingLeftHeight << "," << PaddingLeftWidth << "," << PaddingRightHeight << "," << PaddingRightWidth << "/"
                                    << "Dilation" << DilationHeight << "," << DilationWidth << "/"
                                    << "Stride" << StrideHeight << "," << StrideWidth;
    }

    void MmConv2D(
            size_t GroupCount,
            size_t InputChannels,
            size_t InputHeight,
            size_t InputWidth,
            size_t FilterCount,
            size_t KernelHeight,
            size_t KernelWidth,
            size_t PaddingLeftHeight,
            size_t PaddingLeftWidth,
            size_t PaddingRightHeight,
            size_t PaddingRightWidth,
            size_t DilationHeight,
            size_t DilationWidth,
            size_t StrideHeight,
            size_t StrideWidth,
            size_t OutputHeight,
            size_t OutputWidth,
            const float* Input,
            const float* Filter,
            const float* Bias,
            float* Output)
    {
        shape3d in_shape(GroupCount * InputChannels, InputHeight, InputWidth);
        size_t out_channel = FilterCount * GroupCount;
        bool HasBias = (Bias != nullptr);
        std::vector<size_t> kernel_shape = {KernelHeight, KernelWidth};
        std::vector<size_t> stride_shape = {StrideHeight, StrideWidth};
        std::vector<size_t> dilation_shape = {DilationHeight, DilationWidth};
        xsdnn::padding_mode pad_type = xsdnn::padding_mode::notset;

        std::vector<size_t> pads = {PaddingLeftHeight, PaddingLeftWidth, PaddingRightHeight, PaddingRightWidth};
        params::conv Parameters;
        Parameters._.Dimensions = 2;
        Parameters.infer_output_requirement_shape(in_shape, out_channel, GroupCount, HasBias, kernel_shape,
                                                  stride_shape, dilation_shape, pad_type, pads, MmActivationType::NotSet);
        MmConv(&Parameters._,
               Input,
               Filter, Bias,
               BufferWorking.GetBuffer(Parameters._.TemproraryBufferSize),
               Output);
    }

    void ReferenceConv2D(
            size_t GroupCount,
            size_t InputChannels,
            size_t InputHeight,
            size_t InputWidth,
            size_t FilterCount,
            size_t KernelHeight,
            size_t KernelWidth,
            size_t PaddingLeftHeight,
            size_t PaddingLeftWidth,
            size_t DilationHeight,
            size_t DilationWidth,
            size_t StrideHeight,
            size_t StrideWidth,
            size_t OutputHeight,
            size_t OutputWidth,
            const float* Input,
            const float* Filter,
            const float* Bias,
            float* Output) {
        size_t InputSize = InputHeight * InputWidth;
        size_t OutputSize = OutputHeight * OutputWidth;
        size_t KernelSize = KernelHeight * KernelWidth;

        size_t K = InputChannels * KernelSize;
        size_t Im2ColElements = OutputSize * K;

        const float* filter = Filter;
        const float* bias = Bias;

        for (size_t g = 0; g < GroupCount; g++) {
            //
            // Transform the image using IM2COL and invoke the GEMM.
            //

            float* Im2Col = BufferIm2Col.GetBuffer(Im2ColElements);
            float* Im2ColOut = Im2Col;

            for (size_t c = 0; c < InputChannels; c++) {
                for (size_t ky = 0; ky < KernelHeight; ky++) {
                    for (size_t kx = 0; kx < KernelWidth; kx++) {
                        for (size_t oh = 0; oh < OutputHeight; oh++) {
                            size_t ih = oh * StrideHeight + ky * DilationHeight - PaddingLeftHeight;

                            for (size_t ow = 0; ow < OutputWidth; ow++) {
                                size_t iw = ow * StrideWidth + kx * DilationWidth - PaddingLeftWidth;

                                *Im2ColOut++ = (ih < InputHeight && iw < InputWidth) ? Input[ih * InputWidth + iw] : 0;
                            }
                        }
                    }
                }

                Input += InputSize;
            }

            MmGemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f,
                     filter, K, Im2Col, OutputSize, 0.0f, Output, OutputSize);

            //
            // Apply the bias.
            //

            for (size_t f = 0; f < FilterCount; f++) {
                float biasValue = *bias++;

                for (size_t o = 0; o < OutputSize; o++) {
                    *Output++ += biasValue;
                }
            }

            filter += FilterCount * InputChannels * KernelSize;
        }

    }

public:
    utils::MatrixGuardBuffer<float> BufferInput;
    utils::MatrixGuardBuffer<float> BufferFilter;
    utils::MatrixGuardBuffer<float> BufferBias;
    utils::MatrixGuardBuffer<float> BufferOutput;
    utils::MatrixGuardBuffer<float> BufferOutputReference;
    utils::MatrixGuardBuffer<float> BufferWorking;
    utils::MatrixGuardBuffer<float> BufferIm2Col;
};

int main(int argc, char **argv) {
    SConvTester ConvTest;
//    ConvTest.ExecuteLong();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}