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
#else
#error Not Implemented Yet
#endif

} // mmpack

#endif //XSDNN_DOT_H
