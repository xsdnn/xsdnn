//
// Created by rozhin on 03.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_WRAPPERS_H
#define XSDNN_WRAPPERS_H

#include <cstdint>
#include <type_traits>
#include "macro.h"

#if defined(MM_USE_SSE) && defined(MM_TARGET_AMD64)
#include <immintrin.h>
#endif

namespace mmpack {

#if defined(MM_USE_SSE)

#if !defined(MM_USE_DOUBLE)
typedef __m128 Mm_Float32x4;

template<typename align>
MM_STRONG_INLINE
Mm_Float32x4
MmLoadFloat32x4(const float* Buffer);

template<typename align>
MM_STRONG_INLINE
void
MmStoreFloat32x4(float* Buffer, const Mm_Float32x4& Vector);

template<unsigned pos>
MM_STRONG_INLINE
float
MmExtractPosFloat32x4(const Mm_Float32x4 Vector) {
    return _mm_cvtss_f32(_mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(pos, pos, pos, pos)));
}

MM_STRONG_INLINE
bool
MmIsAligned(const float* x) {
    return reinterpret_cast<uintptr_t>(x) % 16 == 0;
}

MM_STRONG_INLINE
Mm_Float32x4
MmSetZeroFloat32x4(void) {
    return _mm_setzero_ps();
}

MM_STRONG_INLINE
Mm_Float32x4
MmSetOnesFloat32x4(const float& x) {
    return _mm_set1_ps(x);
}

/*
* Load
*/

template<>
MM_STRONG_INLINE
Mm_Float32x4
MmLoadFloat32x4<std::true_type>(const float* Buffer) {
    return _mm_load_ps(Buffer);
}

template<>
MM_STRONG_INLINE
Mm_Float32x4
MmLoadFloat32x4<std::false_type>(const float* Buffer) {
    return _mm_loadu_ps(Buffer);
}


/*
* Store
*/

template<>
MM_STRONG_INLINE
void
MmStoreFloat32x4<std::true_type>(float* Buffer, const Mm_Float32x4& Vector) {
    _mm_store_ps(Buffer, Vector);
}

template<>
MM_STRONG_INLINE
void
MmStoreFloat32x4<std::false_type>(float* Buffer, const Mm_Float32x4& Vector) {
    _mm_storeu_ps(Buffer, Vector);
}

/*
* Unpack Value in Sum
*/
MM_STRONG_INLINE
float
MmUnpackValue(const Mm_Float32x4& Vector) {
    return Vector[0] + Vector[1] + Vector[2] + Vector[3];
}

template<>
MM_STRONG_INLINE
float
MmExtractPosFloat32x4<0>(const Mm_Float32x4 Vector) {
    return _mm_cvtss_f32(Vector);
}

MM_STRONG_INLINE
Mm_Float32x4
MmBroadcastFloat32x4(const float x) {
    return _mm_set1_ps(x);
}


MM_STRONG_INLINE
Mm_Float32x4
MmMultiplyFloat32x4(const Mm_Float32x4& Vector1, const Mm_Float32x4& Vector2) {
    return _mm_mul_ps(Vector1, Vector2);
}

MM_STRONG_INLINE
Mm_Float32x4
MmAddFloat32x4(const Mm_Float32x4& Vector1, const Mm_Float32x4& Vector2) {
    return _mm_add_ps(Vector1, Vector2);
}

/*
 * Multiply Add
 */

MM_STRONG_INLINE
Mm_Float32x4
MmMultiplyAddFloat32x4(const Mm_Float32x4& Vector1, const Mm_Float32x4& Vector2, const Mm_Float32x4& Vector3) {
    return _mm_add_ps(_mm_mul_ps(Vector1, Vector2), Vector3);
}

MM_STRONG_INLINE
Mm_Float32x4
MmMultiplyAddFloat32x4(const Mm_Float32x4& Vector1, const float Value, const Mm_Float32x4& Vector3) {
    return MmMultiplyAddFloat32x4(Vector1, MmBroadcastFloat32x4(Value), Vector3);
}

MM_STRONG_INLINE
Mm_Float32x4
MmMultiplyAddFloat32x4(const Mm_Float32x4& Vector1, const Mm_Float32x4& Vector2, const float Value) {
    return MmMultiplyAddFloat32x4(Vector1, Vector2, MmBroadcastFloat32x4(Value));
}


#else
#error SSE for double type NotImplementedYet
#endif

#elif defined(MM_USE_AVX)
#error AVX Wrappers NotImplementedYet
#endif

} // mmpack

#endif //XSDNN_WRAPPERS_H
