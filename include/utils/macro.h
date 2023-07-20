//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MACRO_H
#define XSDNN_MACRO_H
#include <iostream>

#define XS_UNUSED_PARAMETER(x) (void)(x)
#define XS_2D_1D_CONVERTER(i, j, lda) i * lda + j
#define XS_NUM_THREAD 12

#endif //XSDNN_MACRO_H
