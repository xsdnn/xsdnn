//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_DOT_H
#define XSDNN_DOT_H

#include <cstdlib>
#include "config.h"

namespace mmpack {

#if !defined(MM_USE_DOUBLE)
float
MmDot(
        const float* A,
        const float* B,
        size_t size
);
#else
#error Not Implemented Yet
#endif

} // mmpack

#endif //XSDNN_DOT_H
