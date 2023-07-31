//
// Created by rozhin on 31.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MULADD_H
#define XSDNN_MULADD_H

#include <cstdlib>
#include "config.h"

namespace mmpack {

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

} // mmpack

#endif //XSDNN_MULADD_H
