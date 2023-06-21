//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../gemm.h"
#include "../utils/macro.h"

namespace mmpack {

void ReferenceGemm(
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
) {
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
        if (TransB == CBLAS_TRANSPOSE::CblasNoTrans) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + (m * lda);
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        b += ldb;
                        a += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        } else {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + (m * lda);
                    const float* b = B + (n * ldb);
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        a += 1;
                        b += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        }
    } else {
        if (TransB == CBLAS_TRANSPOSE::CblasNoTrans) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + m;
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        a += lda;
                        b += ldb;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        } else {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    const float* a = A + m;
                    const float* b = B + n * ldb;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; ++k) {
                        sum += (*a * *b);
                        a += lda;
                        b += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        }
    }
}

MM_STRONG_INLINE
void
MmGemmCopyBufferB(
    float* dst,
    const float* src,
    size_t ldb,
    size_t ColNum,
    size_t RowNum
)
/*++

Описание процедуры:

    Копирование матрицы \ подматрицы B в упакованный буфер, с использованием SSE / AVX инструкций.

    Аргументы:

    dst - указатель на буфер назначения.

    src - указатель на матрицу B.

    ldb - лидирующее измерение матрицы В. Равно кол-во столбцов.

    ColNum - кол-во столбцов в матрице B.

    RowNum - кол-во строк в матрице В.

Return Value:

    None.

--*/
{
    bool BIsAligned;
    while(ColNum > 16) {
        size_t y = RowNum;

        Mm_Float32x4 t0;
        Mm_Float32x4 t1;
        Mm_Float32x4 t2;
        Mm_Float32x4 t3;

        const float* b = src;

        do {
            BIsAligned = MmIsAligned(src);

            if (BIsAligned) {
                t0 = MmLoadFloat32x4<std::true_type>(b);
                t1 = MmLoadFloat32x4<std::true_type>(b + 4);
                t2 = MmLoadFloat32x4<std::true_type>(b + 8);
                t3 = MmLoadFloat32x4<std::true_type>(b + 12);
            } else {
                t0 = MmLoadFloat32x4<std::false_type>(b);
                t1 = MmLoadFloat32x4<std::false_type>(b + 4);
                t2 = MmLoadFloat32x4<std::false_type>(b + 8);
                t3 = MmLoadFloat32x4<std::false_type>(b + 12);
            }

            MmStoreFloat32x4<std::true_type>(dst, t0);
            MmStoreFloat32x4<std::true_type>(dst + 4, t1);
            MmStoreFloat32x4<std::true_type>(dst + 8, t2);
            MmStoreFloat32x4<std::true_type>(dst + 12, t3);

            b += ldb;
            dst += 16;
            --y;
        } while (y > 0);

        src += 16;
        ColNum -= 16;
    }

    if (ColNum > 0) {
        Mm_Float32x4 ZeroFloat32x4 = MmSetZeroFloat32x4();

        size_t y = RowNum;

        do {
            float* d = dst;
            const float* b = src;
            BIsAligned = MmIsAligned(src);

            MmStoreFloat32x4<std::true_type>(dst, ZeroFloat32x4);
            MmStoreFloat32x4<std::true_type>(dst + 4, ZeroFloat32x4);
            MmStoreFloat32x4<std::true_type>(dst + 8, ZeroFloat32x4);
            MmStoreFloat32x4<std::true_type>(dst + 12, ZeroFloat32x4);

            if ((ColNum & 8) != 0) {
                Mm_Float32x4 t0;
                Mm_Float32x4 t1;

                if (BIsAligned) {
                    t0 = MmLoadFloat32x4<std::true_type>(b);
                    t1 = MmLoadFloat32x4<std::true_type>(b + 4);
                } else {
                    t0 = MmLoadFloat32x4<std::false_type>(b);
                    t1 = MmLoadFloat32x4<std::false_type>(b + 4);
                }

                MmStoreFloat32x4<std::true_type>(d, t0);
                MmStoreFloat32x4<std::true_type>(d + 4, t1);

                d += 8;
                b += 8;
            }

            if ((ColNum & 4) != 0) {
                Mm_Float32x4 t0;

                if (BIsAligned) {
                    t0 = MmLoadFloat32x4<std::true_type>(b);
                } else {
                    t0 = MmLoadFloat32x4<std::false_type>(b);
                }

                MmStoreFloat32x4<std::true_type>(d, t0);

                d += 4;
                b += 4;
            }

            if ((ColNum & 2) != 0) {
                float t0 = b[0];
                float t1 = b[1];

                d[0] = t0;
                d[1] = t1;

                d += 2;
                b += 2;
            }

            if ((ColNum & 1) != 0) {
                float t0 = b[0];

                d[0] = t0;
            }

            dst += 16;
            src += ldb;
            --y;

        } while (y > 0);
    }
}

MM_STRONG_INLINE
void
MmGemmOp(
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
)
/*++

Описание процедуры:

    C := alpha * op(A) * op(B) + beta * C

    Управляющий метод для матричного умножения, который вызывает другие подфункции
    и управляет указателями на массивы.

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
{
    MM_MAKE_ALIGN(float BufferB[MM_SGEMM_STRIDE_N * MM_SGEMM_STRIDE_K], 16 * sizeof(float));

    /*
     * Оптимизируем размеры шагов, для лучшей утилизации данных в BufferB.
     *
     * Если столбцов в матрице больше, чем строк, то в BufferB должны попадать строки целиком, а не частично.
     *
     * В противном случае, при условии, что матрица А не транспонируется - в BufferB должны попасть столбцы целиком,
     * а не частично.
     */

    size_t StrideN = MM_SGEMM_STRIDE_N;
    size_t StrideK = MM_SGEMM_STRIDE_K;

    if (N >= K) {
        while (StrideK / 2 > K) {
            StrideN *= 2;
            StrideK /= 2;
        }
    } else if (TransA == CblasNoTrans) {
        while (StrideN > 16 && StrideN / 2 > N) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    size_t CountN;

    for (size_t n = 0; n < N; n += CountN) {
        size_t n_temp = N - n;
        CountN = n_temp < StrideN ? n_temp : StrideN;

        if (beta != 0.0f && beta != 1.0f) {
            // FIXME: реализовать домножение
        }

        size_t CountK;
        bool ZeroMode = (beta == 0.0f);

        for (size_t k = 0; k < K; k += CountK) {
            size_t k_temp = K - k;
            CountK = k_temp < StrideK ? k_temp : StrideK;

            /*
             * Выполняем op(B) при необходимости.
             */

            if (TransB == CblasNoTrans) {
                MmGemmCopyBufferB(BufferB, B + n + k * ldb, ldb, CountN, CountK);
            } else {
                // FIXME: реализовать транспонирование матрицы А
            }


        }
    }
}

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
) {
    MmGemmOp(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
} // mmpack

