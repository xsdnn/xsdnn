//
// Created by rozhin on 21.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MMPACK_H
#define XSDNN_MMPACK_H

#include <cstddef>
#include <cstdint>
#include <exception>

#if defined(__GNUC__) || defined(__clang__) || defined(__ICC)
#define MM_STRONG_INLINE __attribute__((always_inline)) inline
#else
#define MM_STRONG_INLINE inline
#endif

// defined SSE support

#if defined (_M_AMD64) || defined (__x86_64)
#define MM_TARGET_AMD64
#else
#error Unsupported system
#endif

#ifndef SSE_INSTR_SET
#if defined ( __AVX2__ )
#define SSE_INSTR_SET 8
#elif defined ( __AVX__ )
#define SSE_INSTR_SET 7
#elif defined ( __SSE4_2__ )
#define SSE_INSTR_SET 6
#elif defined ( __SSE4_1__ )
#define SSE_INSTR_SET 5
#elif defined ( __SSSE3__ )
#define SSE_INSTR_SET 4
#elif defined ( __SSE3__ )
#define SSE_INSTR_SET 3
#elif defined ( __SSE2__ ) || defined ( __x86_64__ )
#define SSE_INSTR_SET 2
    #elif defined ( __SSE__ )
        #define SSE_INSTR_SET 1
    #elif defined ( _M_IX86_FP )
        #define SSE_INSTR_SET _M_IX86_FP
    #else
        #define SSE_INSTR_SET 0
#endif // instruction set defined
#endif // SSE_INSTR_SET

#if SSE_INSTR_SET > 7                  // AVX2 and later
#ifdef __GNUC__
        #include <x86intrin.h>
    #else
        #include <immintrin.h>         // MS version of immintrin.h covers AVX, AVX2 and FMA3
    #endif // __GNUC__
#elif SSE_INSTR_SET == 7
#include <immintrin.h>             // AVX
#elif SSE_INSTR_SET == 6
#include <nmmintrin.h>             // SSE4.2
#elif SSE_INSTR_SET == 5
#include <smmintrin.h>             // SSE4.1
#elif SSE_INSTR_SET == 4
#include <tmmintrin.h>             // SSSE3
#elif SSE_INSTR_SET == 3
#include <pmmintrin.h>             // SSE3
#elif SSE_INSTR_SET == 2
#include <emmintrin.h>             // SSE2
#elif SSE_INSTR_SET == 1
    #include <xmmintrin.h>             // SSE
#endif // SSE_INSTR_SET

// defined aligned memory alloc
#if ((defined __QNXNTO__) || (defined _GNU_SOURCE) || ((defined _XOPEN_SOURCE) && (_XOPEN_SOURCE >= 600))) \
 && (defined _POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO > 0)
#define HAS_POSIX_MEMALIGN 1
#else
#define HAS_POSIX_MEMALIGN 0
#endif

#if SSE_INSTR_SET > 0
#define HAS_MM_MALLOC 1
#else
#define HAS_MM_MALLOC 0
#endif

namespace mmpack {

#ifdef MM_USE_DOUBLE
    typedef double mm_scalar;
#else
    typedef float mm_scalar;
#endif

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
    typedef enum { CblasNoTrans=111, CblasTrans=112 } CBLAS_TRANSPOSE;
#endif

#if !defined(MM_USE_DOUBLE)
float
MmDot(
        const float* A,
        const float* B,
        size_t size
);
/*++

Описание процедуры:

    Вычисление скалярного произведения 2 векторов

    Пример:
         const float* A = {0, 1, 2, 3, 4} // size = 1x5
         const float* B = {0, 1, 2, 3, 4} // size = 1x5

         MmDot(A, B, 5); // 30

    Аргументы:

    A - Задает адресс вектора А.

    B - Задает адресс вектора B.

    size - Размер векторов. Ожидается, что векторы одинакового размера, внутренних проверок нет.

Return Value:

    float Значения скалярного произведения.

--*/
#else
#error Not Implemented Yet
#endif

#if !defined(MM_USE_DOUBLE)
void
MmGemm(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const float* A,
        size_t lda,
        const float* B,
        size_t ldb,
        float beta,
        float* C,
        size_t ldc
);
#else
#error NotImplementedYet
#endif

#if !defined(MM_USE_DOUBLE)
void
MmMulAdd(
        const float* A,
        const float* B,
        float* C,
        size_t size
);
#else
#error NotImplementedYet
#endif

template<typename T, std::size_t alignment>
class aligned_allocator {
public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template <typename U>
    struct rebind {
        typedef aligned_allocator<U, alignment> other;
    };

    aligned_allocator() = default;

    pointer address(reference x) const noexcept;
    const_pointer address(const_reference x) const noexcept;

    pointer allocate(size_type n, const void* hint = 0);
    void deallocate(pointer p, size_type);

    size_type max_size() const noexcept;

    template<class U, class... Args>
    void construct(U* ptr, Args&&... args) {
        void *p = ptr;
        ::new (p) U(std::forward<Args>(args)...);
    }

    template <class U>
    void construct(U *ptr) {
        void *p = ptr;
        ::new (p) U();
    }

    template<class U>
    void destroy(U* ptr) {
        ptr->~U();
    }

private:
    MM_STRONG_INLINE void* aligned_malloc(size_type size);
    MM_STRONG_INLINE void  aligned_free(void* p);
};

template <typename T1, typename T2, std::size_t alignment>
inline bool operator==(const aligned_allocator<T1, alignment> &,
                       const aligned_allocator<T2, alignment> &) {
    return true;
}

template <typename T1, typename T2, std::size_t alignment>
inline bool operator!=(const aligned_allocator<T1, alignment> &,
                       const aligned_allocator<T2, alignment> &) {
    return false;
}

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

template<unsigned Lane>
MM_STRONG_INLINE
void
MmStoreLaneFloat32x4(float* Buffer, const Mm_Float32x4& Vector) {
    _mm_store_ss(Buffer, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
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

MM_STRONG_INLINE
Mm_Float32x4
MmUnpackInterleaveLowFloat32x4(const Mm_Float32x4& Vector1, const Mm_Float32x4& Vector2) {
    return _mm_unpacklo_ps(Vector1, Vector2);
}

MM_STRONG_INLINE
Mm_Float32x4
MmUnpackInterleaveHighFloat32x4(const Mm_Float32x4& Vector1, const Mm_Float32x4& Vector2) {
    return _mm_unpackhi_ps(Vector1, Vector2);
}


#else
#error SSE for double type NotImplementedYet
#endif

#elif defined(MM_USE_AVX)
#error AVX Wrappers NotImplementedYet
#endif


} // mmpack

#endif //XSDNN_MMPACK_H
