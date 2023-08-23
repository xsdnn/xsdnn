//
// Created by rozhin on 21.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MMPACK__H
#define XSDNN_MMPACK__H

#include <mmpack/mmpack.h>

#define MM_UNUSED_PARAMETER(x) (void) (x)

/*
 * Выделяет выровненную память по границе alignment для переменной variable.
 */
#define MM_MAKE_ALIGN(variable, alignment) variable __attribute__ ((aligned(alignment)))


/*
 * Шаги для среза матриц по умолчанию
 */

#define MM_SGEMM_STRIDE_K       128
#define MM_SGEMM_STRIDE_N       128
#define MM_SGEMM_TRANSA_ROWS    12

namespace mmpack {

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

} // mmpack





#endif //XSDNN_MMPACK__H
