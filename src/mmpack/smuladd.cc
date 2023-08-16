//
// Created by rozhin on 31.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <mmpack/muladd.h>
#include <mmpack/macro.h>
#include <mmpack/wrappers.h>

namespace mmpack {

#if !defined(MM_USE_SSE)
MM_STRONG_INLINE
void
ReferenceMulAdd(
        const float* A,
        const float* B,
        float* C,
        size_t size
        )
{
    for (size_t i = 0; i < size; ++i) {
        *C += *A * *B;

        A += 1;
        B += 1;
        C += 1;
    }
}
#else

MM_STRONG_INLINE
void
MmMulAddCopyBuffer(
        const float* src,
        float* dst,
        size_t size
)
/*++

Описание процедуры:

    Копирование вектора src в упакованный буфер dst, с использованием SSE / AVX инструкций.

    Аргументы:

    dst - указатель на буфер назначения.

    src - указатель на исходный вектор.

    size - размер вектора src.

Return Value:

    None.

--*/
{
    size_t Count = size;
    bool srcIsAligned;

    Mm_Float32x4 t0;
    Mm_Float32x4 t1;
    Mm_Float32x4 t2;
    Mm_Float32x4 t3;

    if (Count >= 16) {
        do {
            srcIsAligned = MmIsAligned(src);

            if (srcIsAligned) {
                t0 = MmLoadFloat32x4<std::true_type>(src + 0);
                t1 = MmLoadFloat32x4<std::true_type>(src + 4);
                t2 = MmLoadFloat32x4<std::true_type>(src + 8);
                t3 = MmLoadFloat32x4<std::true_type>(src + 12);
            } else {
                t0 = MmLoadFloat32x4<std::false_type>(src + 0);
                t1 = MmLoadFloat32x4<std::false_type>(src + 4);
                t2 = MmLoadFloat32x4<std::false_type>(src + 8);
                t3 = MmLoadFloat32x4<std::false_type>(src + 12);
            }

            MmStoreFloat32x4<std::true_type>(dst, t0);
            MmStoreFloat32x4<std::true_type>(dst + 4, t1);
            MmStoreFloat32x4<std::true_type>(dst + 8, t2);
            MmStoreFloat32x4<std::true_type>(dst + 12, t3);

            src += 16;
            dst += 16;
            Count -= 16;

        } while (Count >= 16);
    }

    if (Count > 0) {
        Mm_Float32x4 ZeroFloat32x4 = MmSetZeroFloat32x4();

        do {
            srcIsAligned = MmIsAligned(src);

            MmStoreFloat32x4<std::true_type>(dst, ZeroFloat32x4);
            MmStoreFloat32x4<std::true_type>(dst + 4, ZeroFloat32x4);
            MmStoreFloat32x4<std::true_type>(dst + 8, ZeroFloat32x4);
            MmStoreFloat32x4<std::true_type>(dst + 12, ZeroFloat32x4);

            if ((Count & 8) != 0) {

                if (srcIsAligned) {
                    t0 = MmLoadFloat32x4<std::true_type>(src);
                    t1 = MmLoadFloat32x4<std::true_type>(src + 4);
                } else {
                    t0 = MmLoadFloat32x4<std::false_type>(src);
                    t1 = MmLoadFloat32x4<std::false_type>(src + 4);
                }

                MmStoreFloat32x4<std::true_type>(dst, t0);
                MmStoreFloat32x4<std::true_type>(dst + 4, t1);

                dst += 8;
                src += 8;
                Count -= 8;
            }

            if ((Count & 4) != 0) {

                if (srcIsAligned) {
                    t0 = MmLoadFloat32x4<std::true_type>(src);
                } else {
                    t0 = MmLoadFloat32x4<std::false_type>(src);
                }

                MmStoreFloat32x4<std::true_type>(dst, t0);

                dst += 4;
                src += 4;
                Count -= 4;
            }

            if ((Count & 2) != 0) {
                float f0 = src[0];
                float f1 = src[1];

                dst[0] = f0;
                dst[1] = f1;

                dst += 2;
                src += 2;
                Count -= 2;
            }

            if ((Count & 1) != 0) {
                float f0 = src[0];

                dst[0] = f0;
                Count -= 1;
            }

        } while (Count > 0);
    }
}

MM_STRONG_INLINE
void MmMulAddOp(
        const float* A,
        const float* B,
        float* C,
        size_t size
)
/*++

Описание процедуры:

    Операция умножения двух векторов с использованием SSE / AVX инструкций

    Аргументы:

    PackedA - указатель на массив А.

    PackedB - указатель на массив B.

    PackedC - указатель на массив C.

    size - размер массивов.

Return Value:

    None.

--*/
{
    size_t Count = size;
    bool AIsAligned;
    bool BIsAligned;
    bool CIsAligned;

    Mm_Float32x4 A0;
    Mm_Float32x4 A1;
    Mm_Float32x4 A2;
    Mm_Float32x4 A3;

    Mm_Float32x4 B0;
    Mm_Float32x4 B1;
    Mm_Float32x4 B2;
    Mm_Float32x4 B3;

    Mm_Float32x4 C0;
    Mm_Float32x4 C1;
    Mm_Float32x4 C2;
    Mm_Float32x4 C3;

    if (Count >= 16) {
        do {
            AIsAligned = MmIsAligned(A);
            if (AIsAligned) {
                A0 = MmLoadFloat32x4<std::true_type>(A + 0);
                A1 = MmLoadFloat32x4<std::true_type>(A + 4);
                A2 = MmLoadFloat32x4<std::true_type>(A + 8);
                A3 = MmLoadFloat32x4<std::true_type>(A + 12);
             } else {
                A0 = MmLoadFloat32x4<std::false_type>(A + 0);
                A1 = MmLoadFloat32x4<std::false_type>(A + 4);
                A2 = MmLoadFloat32x4<std::false_type>(A + 8);
                A3 = MmLoadFloat32x4<std::false_type>(A + 12);
            }

            BIsAligned = MmIsAligned(B);
            if (BIsAligned) {
                B0 = MmLoadFloat32x4<std::true_type>(B + 0);
                B1 = MmLoadFloat32x4<std::true_type>(B + 4);
                B2 = MmLoadFloat32x4<std::true_type>(B + 8);
                B3 = MmLoadFloat32x4<std::true_type>(B + 12);
            } else {
                B0 = MmLoadFloat32x4<std::false_type>(B + 0);
                B1 = MmLoadFloat32x4<std::false_type>(B + 4);
                B2 = MmLoadFloat32x4<std::false_type>(B + 8);
                B3 = MmLoadFloat32x4<std::false_type>(B + 12);
            }

            CIsAligned = MmIsAligned(C);
            if (CIsAligned) {
                C0 = MmLoadFloat32x4<std::true_type>(C + 0);
                C1 = MmLoadFloat32x4<std::true_type>(C + 4);
                C2 = MmLoadFloat32x4<std::true_type>(C + 8);
                C3 = MmLoadFloat32x4<std::true_type>(C + 12);
            } else {
                C0 = MmLoadFloat32x4<std::false_type>(C + 0);
                C1 = MmLoadFloat32x4<std::false_type>(C + 4);
                C2 = MmLoadFloat32x4<std::false_type>(C + 8);
                C3 = MmLoadFloat32x4<std::false_type>(C + 12);
            }

            if (CIsAligned) {
                MmStoreFloat32x4<std::true_type>(C, MmMultiplyAddFloat32x4(A0, B0, C0));
                MmStoreFloat32x4<std::true_type>(C + 4, MmMultiplyAddFloat32x4(A1, B1, C1));
                MmStoreFloat32x4<std::true_type>(C + 8, MmMultiplyAddFloat32x4(A2, B2, C2));
                MmStoreFloat32x4<std::true_type>(C + 12, MmMultiplyAddFloat32x4(A3, B3, C3));
            } else {
                MmStoreFloat32x4<std::false_type>(C, MmMultiplyAddFloat32x4(A0, B0, C0));
                MmStoreFloat32x4<std::false_type>(C + 4, MmMultiplyAddFloat32x4(A1, B1, C1));
                MmStoreFloat32x4<std::false_type>(C + 8, MmMultiplyAddFloat32x4(A2, B2, C2));
                MmStoreFloat32x4<std::false_type>(C + 12, MmMultiplyAddFloat32x4(A3, B3, C3));
            }

            C += 16;
            A += 16;
            B += 16;
            Count -= 16;

        } while (Count >= 16);
    }

    if (Count > 0) {
        do {
            if ((Count & 8) != 0) {

                AIsAligned = MmIsAligned(A);
                if (AIsAligned) {
                    A0 = MmLoadFloat32x4<std::true_type>(A + 0);
                    A1 = MmLoadFloat32x4<std::true_type>(A + 4);
                } else {
                    A0 = MmLoadFloat32x4<std::false_type>(A + 0);
                    A1 = MmLoadFloat32x4<std::false_type>(A + 4);
                }

                BIsAligned = MmIsAligned(B);
                if (BIsAligned) {
                    B0 = MmLoadFloat32x4<std::true_type>(B + 0);
                    B1 = MmLoadFloat32x4<std::true_type>(B + 4);
                } else {
                    B0 = MmLoadFloat32x4<std::false_type>(B + 0);
                    B1 = MmLoadFloat32x4<std::false_type>(B + 4);
                }

                CIsAligned = MmIsAligned(C);
                if (CIsAligned) {
                    C0 = MmLoadFloat32x4<std::true_type>(C + 0);
                    C1 = MmLoadFloat32x4<std::true_type>(C + 4);
                } else {
                    C0 = MmLoadFloat32x4<std::false_type>(C + 0);
                    C1 = MmLoadFloat32x4<std::false_type>(C + 4);
                }

                if (CIsAligned) {
                    MmStoreFloat32x4<std::true_type>(C, MmMultiplyAddFloat32x4(A0, B0, C0));
                    MmStoreFloat32x4<std::true_type>(C + 4, MmMultiplyAddFloat32x4(A1, B1, C1));
                } else {
                    MmStoreFloat32x4<std::false_type>(C, MmMultiplyAddFloat32x4(A0, B0, C0));
                    MmStoreFloat32x4<std::false_type>(C + 4, MmMultiplyAddFloat32x4(A1, B1, C1));
                }

                A += 8;
                B += 8;
                C += 8;
                Count -= 8;
            }

            if ((Count & 4) != 0) {

                AIsAligned = MmIsAligned(A);
                if (AIsAligned) {
                    A0 = MmLoadFloat32x4<std::true_type>(A + 0);
                } else {
                    A0 = MmLoadFloat32x4<std::false_type>(A + 0);
                }

                BIsAligned = MmIsAligned(B);
                if (BIsAligned) {
                    B0 = MmLoadFloat32x4<std::true_type>(B + 0);
                } else {
                    B0 = MmLoadFloat32x4<std::false_type>(B + 0);
                }

                CIsAligned = MmIsAligned(C);
                if (CIsAligned) {
                    C0 = MmLoadFloat32x4<std::true_type>(C + 0);
                } else {
                    C0 = MmLoadFloat32x4<std::false_type>(C + 0);
                }

                if (CIsAligned) {
                    MmStoreFloat32x4<std::true_type>(C, MmMultiplyAddFloat32x4(A0, B0, C0));
                } else {
                    MmStoreFloat32x4<std::false_type>(C, MmMultiplyAddFloat32x4(A0, B0, C0));
                }

                A += 4;
                B += 4;
                C += 4;
                Count -= 4;
            }

            if ((Count & 2) != 0) {
                float a0 = A[0];
                float a1 = A[1];

                float b0 = B[0];
                float b1 = B[1];

                float c0 = C[0];
                float c1 = C[1];

                C[0] = c0 + a0 * b0;
                C[1] = c1 + a1 * b1;

                A += 2;
                B += 2;
                C += 2;
                Count -= 2;
            }

            if ((Count & 1) != 0) {
                float a0 = A[0];

                float b0 = B[0];

                float c0 = C[0];

                C[0] = c0 + a0 * b0;

                Count -= 1;
            }
        } while (Count > 0);
    }
}

#endif

void
MmMulAdd(
        const float* A,
        const float* B,
        float* C,
        size_t size
) {
#if defined(MM_USE_SSE)
    MmMulAddOp(A, B, C, size);
#else
    ReferenceMulAdd(A, B, C, size);
#endif
}

} // mmpack
