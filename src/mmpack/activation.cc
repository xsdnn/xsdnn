//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <algorithm>
#include "mmpack_.h"

namespace mmpack {

template<MmActivationType ActivationType>
struct MmActivationMain;

template<>
struct MmActivationMain<Relu> {
    Mm_Float32x4 ZeroFloat = MmSetZeroFloat32x4();

    MmActivationMain(const MmActivationHolder* ActivationHolder) {
        MM_UNUSED_PARAMETER(ActivationHolder);
    }

    Mm_Float32x4 Activate(Mm_Float32x4 Vector) {
        return MmMaximumFloat32x4(ZeroFloat, Vector);
    }

    float Activate(float Scalar) {
#ifdef MM_USE_SSE
        return _mm_cvtss_f32(Activate(_mm_set_ss(Scalar)));
#else
        return std::max(Scalar, 0.0f);
#endif // MM_USE_SSE
    }
};

template<>
struct MmActivationMain<HardSigmoid> {
    Mm_Float32x4 Alpha;
    Mm_Float32x4 Beta;
    Mm_Float32x4 Minimum;
    Mm_Float32x4 Maximum;

    MmActivationMain(const MmActivationHolder* ActivationHolder) {
        Alpha = MmBroadcastFloat32x4(ActivationHolder->Parameters.HardSigmoid.alpha);
        Beta = MmBroadcastFloat32x4(ActivationHolder->Parameters.HardSigmoid.beta);
        Minimum = MmSetZeroFloat32x4();
        Maximum = MmBroadcastFloat32x4(1.0f);
    }

    Mm_Float32x4 Activate(Mm_Float32x4 Vector) {
        Vector = MmMultiplyAddFloat32x4(Vector, Alpha, Beta);
        Vector = MmMinimumFloat32x4(Vector, Maximum);
        Vector = MmMaximumFloat32x4(Vector, Minimum);
        return Vector;
    }

    float Activate(float Scalar) {
#if defined(MM_USE_SSE)
        return _mm_cvtss_f32(Activate(_mm_set_ss(Scalar)));
#else
        Scalar = MmExtractPosFloat32x4<0>(Alpha) * Scalar + MmExtractPosFloat32x4<0>(Beta);
        Scalar = std::min(Scalar, MmExtractPosFloat32x4<0>(Maximum));
        Scalar = std::max(Scalar, MmExtractPosFloat32x4<0>(Minimum));
        return Scalar;
#endif
    }
};

template<MmActivationType ActivationType>
void
MmActivationKernel(
    MmActivationHolder* Activation,
    float* C,
    size_t M,
    size_t N,
    size_t ldc
) {
    MmActivationMain<ActivationType> ActivationFunc(Activation);
    bool CIsAligned;
    // Iterate over M rows

    while (M > 0) {
        float* buffer = C;
        size_t n = N;
        CIsAligned = MmIsAligned(buffer);

        // Iterate over N columns
        while (n >= 4) {
            if (CIsAligned) {
                MmStoreFloat32x4<std::true_type>(buffer,
                                                 ActivationFunc.Activate(MmLoadFloat32x4<std::true_type>(buffer)));
            } else {
                MmStoreFloat32x4<std::false_type>(buffer,
                                                 ActivationFunc.Activate(MmLoadFloat32x4<std::false_type>(buffer)));
            }
            buffer += 4;
            n -= 4;
        }

        while (n > 0) {
            *buffer++ = ActivationFunc.Activate(*buffer);
            n -= 1;
        }
        C += ldc;
        M -= 1;
    }
}

void
MmActivation(
        MmActivationHolder* Activation,
        float* C,
        size_t M,
        size_t N,
        size_t ldc
) {
    switch (Activation->ActivationType) {
        case (MmActivationType::Relu):
            MmActivationKernel<Relu>(Activation, C, M, N, ldc);
            break;
        case (MmActivationType::HardSigmoid):
            MmActivationKernel<HardSigmoid>(Activation, C, M, N, ldc);
            break;
        case (NotSet):
            break;
    }
}

void
MmSetDefaultActivationParameters(MmActivationHolder* Holder) {
    Holder->Parameters.HardSigmoid.alpha = 0.2f;
    Holder->Parameters.HardSigmoid.beta = 0.5f;
}

} // mmpack