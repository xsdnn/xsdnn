//
// Created by rozhin on 29.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "mmpack_.h"

namespace mmpack {

void
MmReferenceAdd(
        const float alpha,
        float* C,
        const size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        *C++ = (*C) + alpha;
    }
}

#ifdef MM_TARGET_AMD64
void
MmAddKernelSse(
        const float alpha,
        float* C,
        const size_t size
) {
    size_t Count = size;
    Mm_Float32x4 c0;
    Mm_Float32x4 c1;
    Mm_Float32x4 c2;
    Mm_Float32x4 c3;

    bool CIsAligned;
    Mm_Float32x4 AlphaBroadcast = MmBroadcastFloat32x4(alpha);;

    while (Count >= 16) {
        CIsAligned = MmIsAligned(C);

        if (CIsAligned) {
            c0 = MmLoadFloat32x4<std::true_type>(C + 0);
            c1 = MmLoadFloat32x4<std::true_type>(C + 4);
            c2 = MmLoadFloat32x4<std::true_type>(C + 8);
            c3 = MmLoadFloat32x4<std::true_type>(C + 12);
        } else {
            c0 = MmLoadFloat32x4<std::false_type>(C + 0);
            c1 = MmLoadFloat32x4<std::false_type>(C + 4);
            c2 = MmLoadFloat32x4<std::false_type>(C + 8);
            c3 = MmLoadFloat32x4<std::false_type>(C + 12);
        }

        if (CIsAligned) {
            MmStoreFloat32x4<std::true_type>(C, MmAddFloat32x4(c0, AlphaBroadcast));
            MmStoreFloat32x4<std::true_type>(C + 4, MmAddFloat32x4(c1, AlphaBroadcast));
            MmStoreFloat32x4<std::true_type>(C + 8, MmAddFloat32x4(c2, AlphaBroadcast));
            MmStoreFloat32x4<std::true_type>(C + 12, MmAddFloat32x4(c3, AlphaBroadcast));
        } else {
            MmStoreFloat32x4<std::false_type>(C, MmAddFloat32x4(c0, AlphaBroadcast));
            MmStoreFloat32x4<std::false_type>(C + 4, MmAddFloat32x4(c1, AlphaBroadcast));
            MmStoreFloat32x4<std::false_type>(C + 8, MmAddFloat32x4(c2, AlphaBroadcast));
            MmStoreFloat32x4<std::false_type>(C + 12, MmAddFloat32x4(c3, AlphaBroadcast));
        }

        C += 16;
        Count -= 16;
    }

    if (Count > 0) {
        do {
            if ((Count & 8) != 0) {
                CIsAligned = MmIsAligned(C);

                if (CIsAligned) {
                    c0 = MmLoadFloat32x4<std::true_type>(C + 0);
                    c1 = MmLoadFloat32x4<std::true_type>(C + 4);
                } else {
                    c0 = MmLoadFloat32x4<std::false_type>(C + 0);
                    c1 = MmLoadFloat32x4<std::false_type>(C + 4);
                }

                if (CIsAligned) {
                    MmStoreFloat32x4<std::true_type>(C, MmAddFloat32x4(c0, AlphaBroadcast));
                    MmStoreFloat32x4<std::true_type>(C + 4, MmAddFloat32x4(c1, AlphaBroadcast));
                } else {
                    MmStoreFloat32x4<std::false_type>(C, MmAddFloat32x4(c0, AlphaBroadcast));
                    MmStoreFloat32x4<std::false_type>(C + 4, MmAddFloat32x4(c1, AlphaBroadcast));
                }

                C += 8;
                Count -= 8;
            }

            if ((Count & 4) != 0) {
                CIsAligned = MmIsAligned(C);

                if (CIsAligned) {
                    c0 = MmLoadFloat32x4<std::true_type>(C + 0);
                } else {
                    c0 = MmLoadFloat32x4<std::false_type>(C + 0);
                }

                if (CIsAligned) {
                    MmStoreFloat32x4<std::true_type>(C, MmAddFloat32x4(c0, AlphaBroadcast));
                } else {
                    MmStoreFloat32x4<std::false_type>(C, MmAddFloat32x4(c0, AlphaBroadcast));
                }

                C += 4;
                Count -= 4;
            }

            if ((Count & 2) != 0) {
                C[0] += alpha;
                C[1] += alpha;

                C += 2;
                Count -= 2;
            }

            if ((Count & 1) != 0) {
                C[0] += alpha;

                C += 1;
                Count -= 1;
            }

        } while (Count > 0);
    }
}
#endif
void
MmAdd(
        const float alpha,
        float* C,
        const size_t size
) {
#if defined(MM_TARGET_AMD64)
    MmAddKernelSse(alpha, C, size);
#else
    MmReferenceAdd(alpha, C, size);
#endif
}

}