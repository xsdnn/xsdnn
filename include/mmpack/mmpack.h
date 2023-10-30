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
#warning Unsupported processor. XS Backend Engine will be use reference kernel implementaion.
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

struct MM_CONV_PARAMS {
    enum MmConvAlgorithm {
        Im2ColThenGemm = 0
    };

    size_t Dimensions;
    size_t GroupCount;
    size_t InChannel;
    size_t InShape[2];
    size_t InSize;
    size_t OutShape[2];       // [Cout, Hout, Wout]
    size_t OutSize;
    size_t K;
    size_t Padding[4];         // see onnx.runtime conv op att. pads
    size_t KernelShape[2];
    size_t DilationShape[2];
    size_t StrideShape[2];
    size_t FilterCount;
    MmConvAlgorithm Algorithm;
    bool Bias;
    size_t TemproraryBufferSize;
};
/*++

Описание параметров свертки:

    Dimensions - размерность свертки: поддерживается только 2D. 1D будет добавлена позже.

    GroupCount - кол-во групп, на которые необходимо разбить связи входных и выходных каналов.

    InChannel - кол-во входных каналов в каждой группе.

    InShape - пространственные размеры входной последовательности. 2D: [Hin, Win], 1D: [Win].

    InSize - абсолютная длина входной последовательности.

    OutShape - пространственные размеры выходной последовательности. 2D: [Hout, Wout], 1D: [Wout].

    OutSize - абсолютная длина выходной последовательности.

    K - абсолютная длина всех ядер для каждой группы.

    Padding - кол-во заполнений в формате (y_begin, x_begin, y_end, x_end).

    KernelShape - пространственные размеры ядра.

    DilationShape - пространственные размеры отступов.

    StrideShape - пространственные размеры шага.

    FilterCount - кол-во ядер в каждой группе.

    Algorithm - алгоритм для выполнения свертки.

    Bias - наличие смещения.

    TemprorayBufferSize - размер временного буфера для упаковки результатов Im2Col.
--*/

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
/*++

Описание процедуры:

    C := alpha * op(A) * op(B) + beta * C

Аргументы:

    TransA - транспонировать матрицу А.

    TransB - транспонировать матрицу В.

    M - кол-во строк матрицы А и С.

    N - кол-во столбцов матрицы B и C.

    K - кол-во столбцов матрицы А, кол-во строк матрицы В.

    alpha - коэффициент умножения - см. формулу.

    A - указатель на матрицу A.

    lda - лидирующее измерение матрицы А. Равно кол-во столбцов.

    B - указатель на матрицу В.

    ldb - лидирующее измерение матрицы В. Равно кол-во столбцов.

    beta - коэффициент умножения - см. формулу.

    C - указатель на матрицу C.

    ldc - лидирующее измерение матрицы C. Равно кол-во столбцов.

Return Value:

    None.

--*/

void
MmAdd(
    const float alpha,
    float* C,
    const size_t size
);
/*++

Описание процедуры:

    C := alpha * C

Аргументы:

    alpha - Коэффициент умножения.

    С - указатель на массив.

    size - размер массива C.

Return Value:

    None.

--*/


void
MmMulAdd(
        const float* A,
        const float* B,
        float* C,
        size_t size
);

/*
 * Convolution routines
 */

void
MmConv(
        const MM_CONV_PARAMS* Parameters,
        const float* Input,
        const float* Weight,
        const float* Bias,
        float* TemporaryBuffer,
        float* Output
);
/*++

Описание процедуры:

    Процедура выполняет свертку последовательности. Поддерживается только 2D операция.
    1D операция будет добавлена в следующих версиях.

Аргументы:

    Parameters - контейнер параметров, описывающих процедуру свертки.

    Input - входные данные: одно изображение содержащее C каналов.

    Weight - фильтры для выполнения свертки.

    Bias - опциональное смещение к результату свертки.

    TemporaryBuffer - буфер для результата выполнения алгоритма Im2Col.

    Output - буфер для результата свертки.

Return Value:

    None.

--*/

/*
 * Activation Routines
 */

enum MmActivationType {
    NotSet,
    Relu,
    HardSigmoid
};

struct MmActivationHolder {
    MmActivationType ActivationType;
    union {
        struct {
            float alpha;
            float beta;
        } HardSigmoid;
    } Parameters;
};

void
MmSetDefaultActivationParameters(
        MmActivationHolder* Holder
);

void
MmActivation(
    MmActivationHolder* Activation,
    float* C,
    size_t M,
    size_t N,
    size_t ldc
);
/*++

Описание процедуры:

    Применяет In-Place функцию активации в входному буферу C.

Аргументы:

    Activation - набор параметров для выполнения активации.

    C - входной буффер.

    M - кол-во строк во входном буффере С.

    N - кол-во столбцов во входном буффере С.

    ldc - лидирующее измерение входного буффера С.

Return Value:

    None.

--*/

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


} // mmpack

#endif //XSDNN_MMPACK_H
