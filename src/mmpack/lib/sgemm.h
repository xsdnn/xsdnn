//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <cstdlib>
#include "config.h"

namespace mmpack {

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

} // mmpack
