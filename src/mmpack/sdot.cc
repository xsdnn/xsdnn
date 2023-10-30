//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <type_traits>
#include "mmpack_.h"

namespace mmpack {

float
ReferenceDot(
        const float* A,
        const float* B,
        size_t size
) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        const float* a = A + i;
        const float* b = B + i;
        sum += (*a * *b);
    }
    return sum;
}

#ifdef MM_TARGET_AMD64

template<typename A_aligned, typename B_aligned>
MM_STRONG_INLINE
float
MmDotKernelSse(
        const float* A,
        const float* B,
        size_t size
) {
    Mm_Float32x4 accumulated_vector = MmSetZeroFloat32x4();
    for (size_t i = 0; i < size / 4; i++) {
        Mm_Float32x4 PanelA = MmLoadFloat32x4<A_aligned>(&A[i * 4]);
        Mm_Float32x4 PanelB = MmLoadFloat32x4<B_aligned>(&B[i * 4]);
        Mm_Float32x4 MulPanelAB = MmMultiplyFloat32x4(PanelA, PanelB);
        accumulated_vector = MmAddFloat32x4(accumulated_vector, MulPanelAB);
    }

    float result_dot_product = MmUnpackValue(accumulated_vector);

    /*
     * Учтем не поместившиеся в вектор размера 4 значения
     */

    for (size_t i = (size / 4) * 4; i < size; i++) {
        result_dot_product += A[i] * B[i];
    }
    return result_dot_product;
}


float
MmDotOp(
        const float* A,
        const float* B,
        size_t size
) {
    bool A_aligned = MmIsAligned(A);
    bool B_aligned = MmIsAligned(B);

    if (A_aligned) {
        if (B_aligned) {
            return MmDotKernelSse<std::true_type, std::true_type>(A, B, size);
        } else {
            return MmDotKernelSse<std::true_type, std::false_type>(A, B, size);
        }
    } else {
        if (B_aligned) {
            return MmDotKernelSse<std::false_type, std::true_type>(A, B, size);
        } else {
            return MmDotKernelSse<std::false_type, std::false_type>(A, B, size);
        }
    }
}

#endif


float
MmDot(
        const float* A,
        const float* B,
        size_t size
) {
#if defined(MM_TARGET_AMD64)
    return MmDotOp(A, B, size);
#else
    return ReferenceDot(A, B, size);
#endif
}

} // mmpack