//
// Created by rozhin on 23.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "mmpack_.h"

namespace mmpack {

void
MmConvAddBias(
    const float* Bias,
    float* Output,
    size_t M,
    size_t N,
    size_t ldc
)
/*++

Описание процедуры:

   Добавляет смещение к выходному буферу.

Аргументы:

    Bias - указатель на буфер смещения.

    Output - указатель на выходной буфер.

    M - размер буфера смещения и кол-во строк в выходном буфере.

    N - кол-во столбцов в выходном буфере.

    ldc - лидирующее измерение выходного буфера.

Return Value:

    None.

--*/
{
    size_t CountM = 0;
    while (M-- > 0) {
        MmAdd(Bias[CountM++], Output, N);
        Output += ldc;
    }
}

void
MmConvIm2Col(
        const MM_CONV_PARAMS* Parameters,
        const float* Input,
        float* ColumnBuffer,
        size_t k,
        size_t CountK,
        size_t n,
        size_t CountN
)
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t OutputWidth = Parameters->OutShape[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t nx = (n % OutputWidth);
    const size_t ny = (n / OutputWidth);

    const size_t OriginInputX = nx * StrideWidth;
    const size_t OriginInputY = ny * StrideHeight;

    size_t OutputCountX = OutputWidth - nx;

    const size_t InputHeight = Parameters->InShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InSize;

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;

    Input = Input + (k / (KernelHeight * KernelWidth)) * InputSize;

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    for (size_t EndingK = k + CountK; k < EndingK; k++) {

        size_t CountX = OutputCountX;
        size_t InputY = (ky * DilationHeight) + OriginInputY - PaddingLeftY;
        const size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;

        do {

            if (CountX > RemainingN) {
                CountX = RemainingN;
            }

            RemainingN -= CountX;

            if (InputY < InputHeight) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputY * InputWidth];

                do {

                    if (InputX >= InputWidth) {

                        *ColumnBuffer++ = 0;
                        InputX += StrideWidth;
                        CountX--;

                    } else if (StrideWidth == 1) {

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountX) {
                            CountCopyX = CountX;
                        }

                        CountX -= CountCopyX;

                        while (CountCopyX >= 4) {
                            MmStoreFloat32x4<std::false_type>(ColumnBuffer, MmLoadFloat32x4<std::false_type>(&InputRow[InputX]));
                            ColumnBuffer += 4;
                            InputX += 4;
                            CountCopyX -= 4;
                        }

                        while (CountCopyX > 0) {
                            *ColumnBuffer++ = InputRow[InputX++];
                            CountCopyX--;
                        }

                    } else if (InputX + CountX * StrideWidth <= InputWidth) {

                        do {
                            *ColumnBuffer++ = InputRow[InputX];
                            InputX += StrideWidth;
                        } while (--CountX > 0);

                    } else {

                        do {
                            *ColumnBuffer++ = (InputX < InputWidth) ? InputRow[InputX] : 0;
                            InputX += StrideWidth;
                        } while (--CountX > 0);
                    }

                } while (CountX > 0);

            } else {

                //
                // The entire input row is in the padding region.
                //

                Mm_Float32x4 ZeroFloat32x4 = MmSetZeroFloat32x4();

                while (CountX >= 4) {
                    MmStoreFloat32x4<std::false_type>(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer += 4;
                    CountX -= 4;
                }

                while (CountX > 0) {
                    MmStoreLaneFloat32x4<0>(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer++;
                    CountX--;
                }
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

        } while (RemainingN > 0);

        //
        // Advance the kernel indices and advance to the next channel if the
        // entire kernel is complete.
        //

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                Input += InputSize;

                ky = 0;
            }

            kx = 0;
        }
    }
}

void
MmConvOp(
        const MM_CONV_PARAMS* Parameters,
        const float* Input,
        const float* Weights,
        const float* Bias,
        float* Buffer,
        float* Output,
        size_t SegmentStartN,
        size_t SegmentCountN
) {
    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutSize;
    const size_t K = Parameters->K;

    uint32_t StrideN = MM_SGEMM_STRIDE_N;
    uint32_t StrideK = MM_SGEMM_STRIDE_K;

    if (SegmentCountN >= K) {

        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }

    } else {

        while (StrideN > 16 && StrideN / 2 >= SegmentCountN) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    size_t CountN;

    for (size_t n = 0; n < SegmentCountN; n += CountN) {

        CountN = SegmentCountN - n;

        if (CountN > StrideN) {
            CountN = StrideN;
        }

        size_t CountK;
        float beta = 0.0f;
        float *SegmentOutput = Output + SegmentStartN + n;

        for (size_t k = 0; k < K; k += CountK) {

            CountK = K - k;

            if (CountK > StrideK) {
                CountK = StrideK;
            }

            MmConvIm2Col(Parameters, Input, Buffer, k, CountK,
                         SegmentStartN + n, CountN);

            MmGemm(CblasNoTrans, CblasNoTrans, FilterCount, CountN,
                   CountK, 1.0f, Weights + k, K, Buffer, CountN, beta,
                   SegmentOutput, OutputSize);

            beta = 1.0f;
        }

        if (Bias != nullptr) {
            MmConvAddBias(Bias, SegmentOutput, FilterCount, CountN, OutputSize);
        }
    }
}

void
MmConv(
        const MM_CONV_PARAMS* Parameters,
        const float* Input,
        const float* Weight,
        const float* Bias,
        float* TemporaryBuffer,
        float* Output
) {
        const size_t FilterCount = Parameters->FilterCount;
        const size_t OutputSize = Parameters->OutSize;
        const size_t K = Parameters->K;

        const size_t SpatialInputGroupSize = Parameters->InChannel * Parameters->InSize;
        const size_t SpatialOutputGroupSize = OutputSize * FilterCount;
        const size_t FilterGroupSize = FilterCount * K;

        const size_t GroupCount = Parameters->GroupCount;

        const float* filter = Weight;
        const float* bias = Bias;

        for (size_t group = 0; group < GroupCount; ++group) {
            switch (Parameters->Algorithm) {
                case(MM_CONV_PARAMS::Im2ColThenGemm) : {

                    MmConvOp(Parameters, Input, filter, bias, TemporaryBuffer, Output, 0, OutputSize);

                    break;
                }
            }

            if (bias != nullptr) {
                bias += FilterCount;
            }

            filter += FilterGroupSize;
            Input += SpatialInputGroupSize;
            Output += SpatialOutputGroupSize;
        }
}

}