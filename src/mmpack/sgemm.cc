//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "mmpack_.h"

namespace mmpack {

#if !defined(MM_USE_SSE)
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

#else

MM_STRONG_INLINE
void
MmGemmMulBeta(
    float* C,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    float beta
)
/*++

Описание процедуры:

    Домножение выходной матрицы С на коэффициент beta - см. формулу.

Аргументы:

    С - указатель на выходную матрицу.

    CountМ - кол-во строк матрицы С.

    CountN - кол-во столбцов матрицы С.

    ldc - лидирующее измерение матрицы C. Равно кол-во столбцов.

    beta - коэфф. домножения

Return Value:

    None.

--*/
{
    Mm_Float32x4 Beta = MmBroadcastFloat32x4(beta);

    /*
     * Полность обойдем матрицу и домножим на beta.
     */
    size_t n;
    float* c;

    while (CountM > 0) {
        n = CountN;
        c = C;

        while (n >= 4) {
            MmStoreFloat32x4<std::false_type>(c,
                                              MmMultiplyFloat32x4(
                                                    MmLoadFloat32x4<std::false_type>(c),
                                                    Beta));
            c += 4;
            n -= 4;
        }

        while (n > 0) {
            _mm_store_ss(c,
                         _mm_mul_ss(
                                 _mm_load_ss(c),
                                 Beta
                    ));
            c += 1;
            n -= 1;
        }

        C += ldc;
        CountM -= 1;
    }
}

MM_STRONG_INLINE
void
MmGemmTransposeA(
    float* dst,
    const float* src,
    size_t lda,
    size_t ColNum,
    size_t RowNum
)
/*++

Описание процедуры:

    Транспонирование матрицы \ подматрицы A.

Аргументы:

    dst - указатель на буфер назначения.

    src - указатель на матрицу А.

    ldа - лидирующее измерение матрицы А. Равно кол-во столбцов.

    ColNum - кол-во столбцов в матрице А.

    RowNum - кол-во строк в матрице А.

Return Value:

    None.

--*/
{
    size_t lddst = RowNum;

    while (RowNum >= 4) {
        float* d = dst;
        const float* a = src;
        size_t y = ColNum;

        do {
            float t0 = a[0];
            float t1 = a[lda];
            float t2 = a[lda * 2];
            float t3 = a[lda * 3];

            d[0] = t0;
            d[1] = t1;
            d[2] = t2;
            d[3] = t3;

            d += lddst;
            a += 1;
            --y;
        } while (y > 0);

        dst += 4;
        src += lda * 4;
        RowNum -= 4;
    }

    if (RowNum >= 2) {

        float* d = dst;
        const float* a = src;
        size_t y = ColNum;

        do {

            float t0 = a[0];
            float t1 = a[lda];

            d[0] = t0;
            d[1] = t1;

            d += lddst;
            a += 1;
            --y;

        } while (y > 0);

        dst += 2;
        src += lda * 2;
        RowNum -= 2;
    }

    if (RowNum >= 1) {

        float* d = dst;
        const float* a = src;
        size_t y = ColNum;

        do {

            d[0] = a[0];

            d += lddst;
            a += 1;
            --y;

        } while (y > 0);
    }
}

template<unsigned N>
MM_STRONG_INLINE
void
MmGemmTransposeBufferBNx4(
        float* dst,
        const float* src,
        size_t ldb
)
/*++

Описание процедуры:

    Транспонирование подматрицы src размера Nx4.

Аргументы:

    dst - указатель на буфер назначения.

    src - указатель на подматрицу.

    ldb - лидирующее измерение подматрицы.

Return Value:

    None.

--*/
    {
        for (unsigned n = 0; n < N / 4; n++) {

            Mm_Float32x4 t0 = MmLoadFloat32x4<std::false_type>(&src[ldb * 0]);
            Mm_Float32x4 t1 = MmLoadFloat32x4<std::false_type>(&src[ldb * 1]);
            Mm_Float32x4 t2 = MmLoadFloat32x4<std::false_type>(&src[ldb * 2]);
            Mm_Float32x4 t3 = MmLoadFloat32x4<std::false_type>(&src[ldb * 3]);

            Mm_Float32x4 z0 = MmUnpackInterleaveLowFloat32x4(t0, t2);
            Mm_Float32x4 z1 = MmUnpackInterleaveHighFloat32x4(t0, t2);
            Mm_Float32x4 z2 = MmUnpackInterleaveLowFloat32x4(t1, t3);
            Mm_Float32x4 z3 = MmUnpackInterleaveHighFloat32x4(t1, t3);

            t0 = MmUnpackInterleaveLowFloat32x4(z0, z2);
            t1 = MmUnpackInterleaveHighFloat32x4(z0, z2);
            t2 = MmUnpackInterleaveLowFloat32x4(z1, z3);
            t3 = MmUnpackInterleaveHighFloat32x4(z1, z3);

            MmStoreFloat32x4<std::true_type>(&dst[0], t0);
            MmStoreFloat32x4<std::true_type>(&dst[16], t1);
            MmStoreFloat32x4<std::true_type>(&dst[32], t2);
            MmStoreFloat32x4<std::true_type>(&dst[48], t3);

            dst += 4;
            src += ldb * 4;
        }
    }

MM_STRONG_INLINE
void
MmGemmTransposeBufferB(
    float* dst,
    const float* src,
    size_t ldb,
    size_t RowNum,
    size_t ColNum
)
/*++

Описание процедуры:

    Транспонирование и упаковка матрицы \ подматрицы B, с использованием SSE / AVX / FMA инструкций.

Аргументы:

    dst - указатель на буфер назначения.

    src - указатель на матрицу В.

    ldb - лидирующее измерение матрицы B. Равно кол-во столбцов.

    RowNum - кол-во строк в матрице B.

    ColNum - кол-во столбцов в матрице B.

Return Value:

    None.

--*/
    {
        while (RowNum >= 16) {

            const float* b = src;
            size_t x = ColNum;

            while (x >= 4) {

                MmGemmTransposeBufferBNx4<16>(&dst[0], &b[0], ldb);

                dst += 16 * 4;
                b += 4;
                x -= 4;
            }

            while (x > 0) {

                float t0 = b[0];
                float t1 = b[ldb];
                float t2 = b[ldb * 2];
                float t3 = b[ldb * 3];
                float t4 = b[ldb * 4];
                float t5 = b[ldb * 5];
                float t6 = b[ldb * 6];
                float t7 = b[ldb * 7];
                float t8 = b[ldb * 8];
                float t9 = b[ldb * 9];
                float t10 = b[ldb * 10];
                float t11 = b[ldb * 11];
                float t12 = b[ldb * 12];
                float t13 = b[ldb * 13];
                float t14 = b[ldb * 14];
                float t15 = b[ldb * 15];

                dst[0] = t0;
                dst[1] = t1;
                dst[2] = t2;
                dst[3] = t3;
                dst[4] = t4;
                dst[5] = t5;
                dst[6] = t6;
                dst[7] = t7;
                dst[8] = t8;
                dst[9] = t9;
                dst[10] = t10;
                dst[11] = t11;
                dst[12] = t12;
                dst[13] = t13;
                dst[14] = t14;
                dst[15] = t15;

                dst += 16;
                b += 1;
                x--;
            }

            src += ldb * 16;
            RowNum -= 16;
        }

        if (RowNum > 0) {

            Mm_Float32x4 ZeroFloat32x4 = MmSetZeroFloat32x4();

            size_t x = ColNum;

            while (x >= 4) {

                float* d = dst;
                const float* b = src;

                if ((RowNum & 8) != 0) {

                    MmGemmTransposeBufferBNx4<8>(&d[0], &b[0], ldb);

                    d += 8;
                    b += ldb * 8;

                } else {

                    MmStoreFloat32x4<std::true_type>(&d[8], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[12], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[24], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[28], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[40], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[44], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[56], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[60], ZeroFloat32x4);
                }

                if ((RowNum & 4) != 0) {

                    MmGemmTransposeBufferBNx4<4>(&d[0], &b[0], ldb);

                    d += 4;
                    b += ldb * 4;

                } else {

                    MmStoreFloat32x4<std::true_type>(&d[4], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[20], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[36], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[52], ZeroFloat32x4);
                }

                MmStoreFloat32x4<std::true_type>(&d[0], ZeroFloat32x4);
                MmStoreFloat32x4<std::true_type>(&d[16], ZeroFloat32x4);
                MmStoreFloat32x4<std::true_type>(&d[32], ZeroFloat32x4);
                MmStoreFloat32x4<std::true_type>(&d[48], ZeroFloat32x4);

                if ((RowNum & 2) != 0) {

                    Mm_Float32x4 t0 = MmLoadFloat32x4<std::false_type>(&b[0]);
                    Mm_Float32x4 t1 = MmLoadFloat32x4<std::false_type>(&b[ldb]);

                    MmStoreLaneFloat32x4<0>(&d[0], t0);
                    MmStoreLaneFloat32x4<0>(&d[1], t1);
                    MmStoreLaneFloat32x4<1>(&d[16], t0);
                    MmStoreLaneFloat32x4<1>(&d[17], t1);
                    MmStoreLaneFloat32x4<2>(&d[32], t0);
                    MmStoreLaneFloat32x4<2>(&d[33], t1);
                    MmStoreLaneFloat32x4<3>(&d[48], t0);
                    MmStoreLaneFloat32x4<3>(&d[49], t1);

                    d += 2;
                    b += ldb * 2;
                }

                if ((RowNum & 1) != 0) {
                    d[0] = b[0];
                    d[16] = b[1];
                    d[32] = b[2];
                    d[48] = b[3];
                }

                dst += 16 * 4;
                src += 4;
                x -= 4;
            }

            while (x > 0) {

                float* d = dst;
                const float* b = src;

                if ((RowNum & 8) != 0) {

                    float t0 = b[0];
                    float t1 = b[ldb];
                    float t2 = b[ldb * 2];
                    float t3 = b[ldb * 3];
                    float t4 = b[ldb * 4];
                    float t5 = b[ldb * 5];
                    float t6 = b[ldb * 6];
                    float t7 = b[ldb * 7];

                    d[0] = t0;
                    d[1] = t1;
                    d[2] = t2;
                    d[3] = t3;
                    d[4] = t4;
                    d[5] = t5;
                    d[6] = t6;
                    d[7] = t7;

                    d += 8;
                    b += ldb * 8;

                } else {

                    MmStoreFloat32x4<std::true_type>(&d[8], ZeroFloat32x4);
                    MmStoreFloat32x4<std::true_type>(&d[12], ZeroFloat32x4);
                }

                if ((RowNum & 4) != 0) {

                    float t0 = b[0];
                    float t1 = b[ldb];
                    float t2 = b[ldb * 2];
                    float t3 = b[ldb * 3];

                    d[0] = t0;
                    d[1] = t1;
                    d[2] = t2;
                    d[3] = t3;

                    d += 4;
                    b += ldb * 4;

                } else {
                    MmStoreFloat32x4<std::true_type>(&d[4], ZeroFloat32x4);
                }

                MmStoreFloat32x4<std::true_type>(d, ZeroFloat32x4);

                if ((RowNum & 2) != 0) {

                    float t0 = b[0];
                    float t1 = b[ldb];

                    d[0] = t0;
                    d[1] = t1;

                    d += 2;
                    b += ldb * 2;
                }

                if ((RowNum & 1) != 0) {
                    d[0] = b[0];
                }

                dst += 16;
                src += 1;
                x--;
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
    while(ColNum >= 16) {
        size_t y = RowNum;

        Mm_Float32x4 t0;
        Mm_Float32x4 t1;
        Mm_Float32x4 t2;
        Mm_Float32x4 t3;

        const float* b = src;

        do {
            BIsAligned = MmIsAligned(b);

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
            BIsAligned = MmIsAligned(b);

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

template<bool ZeroMode, bool ProcessTwoRows>
MM_STRONG_INLINE
size_t
MmGemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    float alpha
)
/*++

Описание процедуры:

    Ядро умножения - умножение одной или двух строк за операцию.
    Поддерживает работу с занулением или добавлением содержимого к матрице С.

    Аргументы:

    A - указатель на матрицу A.

    B - указатель на упакованный буфер В.

    C - указатель на матрицу C.

    CountN - кол-во столбцов матрицы B и C для обработк.

    CountK - кол-во столбцов матрицы А, кол-во строк матрицы В для обработки.

    lda - лидирующее измерение матрицы А. Равно кол-во столбцов.

    ldc - лидирующее измерение матрицы C. Равно кол-во столбцов.

    alpha - коэффициент умножения - см. формулу.

Return Value:

    кол-во обработанных (умноженных) строк.

--*/
{
    Mm_Float32x4 r0_b0; // row 0, block 0 у матрицы C.
    Mm_Float32x4 r0_b1;
    Mm_Float32x4 r0_b2;
    Mm_Float32x4 r0_b3;

    Mm_Float32x4 r1_b0;
    Mm_Float32x4 r1_b1;
    Mm_Float32x4 r1_b2;
    Mm_Float32x4 r1_b3;

    // Бродкастим значение коэфф. alpha для дальнейшего умножения.

    Mm_Float32x4 Alpha = MmBroadcastFloat32x4(alpha);

    bool BIsAligned;

    do {

        Mm_Float32x4 B_e0; // Element B[i * 4]
        Mm_Float32x4 B_e1;
        Mm_Float32x4 B_e2;
        Mm_Float32x4 B_e3;

        float r0A_e0; // row 0, element 0 у матрицы А.
        float r0A_e1;
        float r1A_e0;
        float r1A_e1;

        /*
         * Зануляем значения, с которыми будем работать при умножении.
         */
        r0_b0 = MmSetZeroFloat32x4();
        r0_b1 = MmSetZeroFloat32x4();
        r0_b2 = MmSetZeroFloat32x4();
        r0_b3 = MmSetZeroFloat32x4();

        if (ProcessTwoRows) {
            r1_b0 = MmSetZeroFloat32x4();
            r1_b1 = MmSetZeroFloat32x4();
            r1_b2 = MmSetZeroFloat32x4();
            r1_b3 = MmSetZeroFloat32x4();
        }

        /*
         * Вычисляем блок 1х16 или 2х16
         */

        const float* a = A;
        size_t k = CountK;

        while (k >= 2) {

            BIsAligned = MmIsAligned(B);

            r0A_e0 = a[0];
            r0A_e1 = a[1];

            if (ProcessTwoRows) {
                r1A_e0 = a[lda];
                r1A_e1 = a[lda + 1];
            }

            if (BIsAligned) {
                B_e0 = MmLoadFloat32x4<std::true_type>(B);
                B_e1 = MmLoadFloat32x4<std::true_type>(B + 4);
                B_e2 = MmLoadFloat32x4<std::true_type>(B + 8);
                B_e3 = MmLoadFloat32x4<std::true_type>(B + 12);
            } else {
                B_e0 = MmLoadFloat32x4<std::false_type>(B);
                B_e1 = MmLoadFloat32x4<std::false_type>(B + 4);
                B_e2 = MmLoadFloat32x4<std::false_type>(B + 8);
                B_e3 = MmLoadFloat32x4<std::false_type>(B + 12);
            }

            r0_b0 = MmMultiplyAddFloat32x4(B_e0, r0A_e0, r0_b0);
            r0_b1 = MmMultiplyAddFloat32x4(B_e1, r0A_e0, r0_b1);
            r0_b2 = MmMultiplyAddFloat32x4(B_e2, r0A_e0, r0_b2);
            r0_b3 = MmMultiplyAddFloat32x4(B_e3, r0A_e0, r0_b3);

            if (ProcessTwoRows) {
                r1_b0 = MmMultiplyAddFloat32x4(B_e0, r1A_e0, r1_b0);
                r1_b1 = MmMultiplyAddFloat32x4(B_e1, r1A_e0, r1_b1);
                r1_b2 = MmMultiplyAddFloat32x4(B_e2, r1A_e0, r1_b2);
                r1_b3 = MmMultiplyAddFloat32x4(B_e3, r1A_e0, r1_b3);
            }


            if (BIsAligned) {
                B_e0 = MmLoadFloat32x4<std::true_type>(B + 16);
                B_e1 = MmLoadFloat32x4<std::true_type>(B + 20);
                B_e2 = MmLoadFloat32x4<std::true_type>(B + 24);
                B_e3 = MmLoadFloat32x4<std::true_type>(B + 28);
            } else {
                B_e0 = MmLoadFloat32x4<std::false_type>(B + 16);
                B_e1 = MmLoadFloat32x4<std::false_type>(B + 20);
                B_e2 = MmLoadFloat32x4<std::false_type>(B + 24);
                B_e3 = MmLoadFloat32x4<std::false_type>(B + 28);
            }

            r0_b0 = MmMultiplyAddFloat32x4(B_e0, r0A_e1, r0_b0);
            r0_b1 = MmMultiplyAddFloat32x4(B_e1, r0A_e1, r0_b1);
            r0_b2 = MmMultiplyAddFloat32x4(B_e2, r0A_e1, r0_b2);
            r0_b3 = MmMultiplyAddFloat32x4(B_e3, r0A_e1, r0_b3);

            if (ProcessTwoRows) {
                r1_b0 = MmMultiplyAddFloat32x4(B_e0, r1A_e1, r1_b0);
                r1_b1 = MmMultiplyAddFloat32x4(B_e1, r1A_e1, r1_b1);
                r1_b2 = MmMultiplyAddFloat32x4(B_e2, r1A_e1, r1_b2);
                r1_b3 = MmMultiplyAddFloat32x4(B_e3, r1A_e1, r1_b3);
            }

            a += 2;
            B += 32;
            k -= 2;
        }

        if (k > 0) {

            BIsAligned = MmIsAligned(B);

            r0A_e0 = a[0];

            if (ProcessTwoRows) {
                r1A_e0 = a[lda];
            }

            if (BIsAligned) {
                B_e0 = MmLoadFloat32x4<std::true_type>(B + 0);
                B_e1 = MmLoadFloat32x4<std::true_type>(B + 4);
                B_e2 = MmLoadFloat32x4<std::true_type>(B + 8);
                B_e3 = MmLoadFloat32x4<std::true_type>(B + 12);
            } else {
                B_e0 = MmLoadFloat32x4<std::false_type>(B + 0);
                B_e1 = MmLoadFloat32x4<std::false_type>(B + 4);
                B_e2 = MmLoadFloat32x4<std::false_type>(B + 8);
                B_e3 = MmLoadFloat32x4<std::false_type>(B + 12);
            }

            r0_b0 = MmMultiplyAddFloat32x4(B_e0, r0A_e0, r0_b0);
            r0_b1 = MmMultiplyAddFloat32x4(B_e1, r0A_e0, r0_b1);
            r0_b2 = MmMultiplyAddFloat32x4(B_e2, r0A_e0, r0_b2);
            r0_b3 = MmMultiplyAddFloat32x4(B_e3, r0A_e0, r0_b3);

            if (ProcessTwoRows) {
                r1_b0 = MmMultiplyAddFloat32x4(B_e0, r1A_e0, r1_b0);
                r1_b1 = MmMultiplyAddFloat32x4(B_e1, r1A_e0, r1_b1);
                r1_b2 = MmMultiplyAddFloat32x4(B_e2, r1A_e0, r1_b2);
                r1_b3 = MmMultiplyAddFloat32x4(B_e3, r1A_e0, r1_b3);
            }

            B += 16;
        }

        /*
         * Домножим значения на alpha
         */

        r0_b0 = MmMultiplyFloat32x4(r0_b0, Alpha);
        r0_b1 = MmMultiplyFloat32x4(r0_b1, Alpha);
        r0_b2 = MmMultiplyFloat32x4(r0_b2, Alpha);
        r0_b3 = MmMultiplyFloat32x4(r0_b3, Alpha);

        if (ProcessTwoRows) {
            r1_b0 = MmMultiplyFloat32x4(r1_b0, Alpha);
            r1_b1 = MmMultiplyFloat32x4(r1_b1, Alpha);
            r1_b2 = MmMultiplyFloat32x4(r1_b2, Alpha);
            r1_b3 = MmMultiplyFloat32x4(r1_b3, Alpha);
        }

        /*
         * Сохраним значения в результирующую матрицу С целиком по 16 значений.
         */

        if (CountN >= 16) {

            if (!ZeroMode) {
                r0_b0 = MmAddFloat32x4(r0_b0, MmLoadFloat32x4<std::false_type>(C + 0));
                r0_b1 = MmAddFloat32x4(r0_b1, MmLoadFloat32x4<std::false_type>(C + 4));
                r0_b2 = MmAddFloat32x4(r0_b2, MmLoadFloat32x4<std::false_type>(C + 8));
                r0_b3 = MmAddFloat32x4(r0_b3, MmLoadFloat32x4<std::false_type>(C + 12));
            }

            MmStoreFloat32x4<std::false_type>(C, r0_b0);
            MmStoreFloat32x4<std::false_type>(C + 4, r0_b1);
            MmStoreFloat32x4<std::false_type>(C + 8, r0_b2);
            MmStoreFloat32x4<std::false_type>(C + 12, r0_b3);


            if (ProcessTwoRows) {

                if (!ZeroMode) {
                    r1_b0 = MmAddFloat32x4(r1_b0, MmLoadFloat32x4<std::false_type>(C + ldc + 0));
                    r1_b1 = MmAddFloat32x4(r1_b1, MmLoadFloat32x4<std::false_type>(C + ldc + 4));
                    r1_b2 = MmAddFloat32x4(r1_b2, MmLoadFloat32x4<std::false_type>(C + ldc + 8));
                    r1_b3 = MmAddFloat32x4(r1_b3, MmLoadFloat32x4<std::false_type>(C + ldc + 12));
                }

                MmStoreFloat32x4<std::false_type>(C + ldc + 0, r1_b0);
                MmStoreFloat32x4<std::false_type>(C + ldc + 4, r1_b1);
                MmStoreFloat32x4<std::false_type>(C + ldc + 8, r1_b2);
                MmStoreFloat32x4<std::false_type>(C + ldc + 12, r1_b3);
            }

        } else {
            /*
             * Сохраним значения в результирующую матрицу С частично.
             */

            if ((CountN & 8) != 0) {

                if (!ZeroMode) {
                    r0_b0 = MmAddFloat32x4(r0_b0, MmLoadFloat32x4<std::false_type>(C + 0));
                    r0_b1 = MmAddFloat32x4(r0_b1, MmLoadFloat32x4<std::false_type>(C + 4));
                }

                MmStoreFloat32x4<std::false_type>(C + 0, r0_b0);
                MmStoreFloat32x4<std::false_type>(C + 4, r0_b1);


                r0_b0 = r0_b2;
                r0_b1 = r0_b3;

                if (ProcessTwoRows) {


                    if (!ZeroMode) {
                        r1_b0 = MmAddFloat32x4(r1_b0, MmLoadFloat32x4<std::false_type>(C + ldc + 0));
                        r1_b1 = MmAddFloat32x4(r1_b1, MmLoadFloat32x4<std::false_type>(C + ldc + 4));
                    }

                    MmStoreFloat32x4<std::false_type>(C + ldc + 0, r1_b0);
                    MmStoreFloat32x4<std::false_type>(C + ldc + 4, r1_b1);


                    r1_b0 = r1_b2;
                    r1_b1 = r1_b3;
                }

                C += 8;
            }

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    r0_b0 = MmAddFloat32x4(r0_b0, MmLoadFloat32x4<std::false_type>(C + 0));
                }

                MmStoreFloat32x4<std::false_type>(C + 0, r0_b0);


                r0_b0 = r0_b1;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        r1_b0 = MmAddFloat32x4(r1_b0, MmLoadFloat32x4<std::false_type>(C + ldc + 0));
                    }

                    MmStoreFloat32x4<std::false_type>(C + ldc + 0, r1_b0);

                    r1_b0 = r1_b1;
                }

                C += 4;

            }

            float r0_b00 = MmExtractPosFloat32x4<0>(r0_b0);
            float r0_b01 = MmExtractPosFloat32x4<1>(r0_b0);
            float r1_b00;
            float r1_b01;

            if (ProcessTwoRows) {
                r1_b00 = MmExtractPosFloat32x4<0>(r1_b0);
                r1_b01 = MmExtractPosFloat32x4<1>(r1_b0);
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    r0_b00 += C[0];
                    r0_b01 += C[1];
                }

                C[0] = r0_b00;
                C[1] = r0_b01;
                r0_b00 = MmExtractPosFloat32x4<2>(r0_b0);
                r0_b01 = MmExtractPosFloat32x4<3>(r0_b0);

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        r1_b00 += C[ldc + 0];
                        r1_b01 += C[ldc + 1];
                    }

                    C[ldc + 0] = r1_b00;
                    C[ldc + 1] = r1_b01;
                    r1_b00 = MmExtractPosFloat32x4<2>(r1_b0);
                    r1_b01 = MmExtractPosFloat32x4<3>(r1_b0);

                }

                C += 2;
            }

            if ((CountN & 1) != 0) {

                if (!ZeroMode) {
                    r0_b00 += C[0];
                }

                C[0] = r0_b00;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        r1_b00 += C[ldc + 0];
                    }

                    C[ldc + 0] = r1_b00;

                }

                C += 2;

            }

            break;
        }

        C += 16;
        CountN -= 16;

    } while (CountN > 0);

    return ProcessTwoRows ? 2 : 1;
}

MM_STRONG_INLINE
float*
MmGemmKernelLoop(
    const float* A,
    const float* B,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
)
/*++

Описание процедуры:

    Построчно (по строкам матрицы С) производит умножение.

    Аргументы:

    A - указатель на матрицу A.

    B - указатель на упакованный буфер В.

    C - указатель на матрицу C.

    CountM - кол-во строк матрицы А и С для обработки.

    CountN - кол-во столбцов матрицы B и C для обработк.

    CountK - кол-во столбцов матрицы А, кол-во строк матрицы В для обработки.

    lda - лидирующее измерение матрицы А. Равно кол-во столбцов.

    ldc - лидирующее измерение матрицы C. Равно кол-во столбцов.

    alpha - коэффициент умножения - см. формулу.

    ZeroMode - перезаписывать значения в матрице C?

Return Value:

    указатель на начало необработанной части матрицы С.

--*/
{
    size_t RowsProcessed;
    while (CountM > 0) {
        if (ZeroMode) {

            if (CountM >= 2) {
                RowsProcessed = MmGemmKernel<true, true>(A, B, C, CountN, CountK, lda, ldc, alpha);
            } else {
                RowsProcessed = MmGemmKernel<true, false>(A, B, C, CountN, CountK, lda, ldc, alpha);
            }

        } else {

            if (CountM >= 2) {
                RowsProcessed = MmGemmKernel<false, true>(A, B, C, CountN, CountK, lda, ldc, alpha);
            } else {
                RowsProcessed = MmGemmKernel<false, false>(A, B, C, CountN, CountK, lda, ldc, alpha);
            }

        }

        C += ldc * RowsProcessed;
        A += lda * RowsProcessed;
        CountM -= RowsProcessed;
    }
    return C;
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
    float BufferA[MM_SGEMM_TRANSA_ROWS * MM_SGEMM_STRIDE_K];
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
            MmGemmMulBeta(C + n, M, CountN, ldc, beta);
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
                MmGemmTransposeBufferB(BufferB, B + k + n * ldb, ldb, CountN, CountK);
            }

            float* c = C + n;

            if (TransA == CblasNoTrans) {
                MmGemmKernelLoop(A + k, BufferB, c, M, CountN, CountK, lda, ldc, alpha, ZeroMode);
            } else {
                const float* a = A + k * lda;
                size_t RowsProcessed = M;

                while (RowsProcessed > 0) {
                    size_t RowsTransposed = RowsProcessed > size_t(MM_SGEMM_TRANSA_ROWS)
                                            ? size_t(MM_SGEMM_TRANSA_ROWS) : RowsProcessed;

                    MmGemmTransposeA(BufferA, a, lda, RowsTransposed, CountK);

                    RowsProcessed -= RowsTransposed;
                    a += RowsTransposed;

                    c = MmGemmKernelLoop(BufferA, BufferB, c, RowsTransposed, CountN, CountK, CountK, ldc, alpha, ZeroMode);
                }
            }
            ZeroMode = false;
        }
    }
}
#endif

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
#if defined(MM_USE_SSE)
    MmGemmOp(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    ReferenceGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}
} // mmpack

