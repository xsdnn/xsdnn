//
// Created by rozhin on 07.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

/*
 * В этом модуле имплементирована операция MmVectorToMatrixMul
 *
 * Эквивалент операции Trans NoTrans GEMM, где два операнда - строковые векторы (rows count == 1)
 */


#ifndef XSDNN_SV2MM_H
#define XSDNN_SV2MM_H

#include <cstdlib>
#include "config.h"

namespace mmpack {

#if !defined(MM_USE_DOUBLE)
void
MmVectorToMatrixMul(
        size_t K,
        size_t N,
        float alpha,
        const float* A,
        const float* B,
        float beta,
        float* C,
        size_t ldc
);
/*++

Описание процедуры:

    Вычисление Vector/Vector -> Matrix умножения.

    Пример:
         mat_t A = {0, 1, 2, 3, 4} // size = 1x5
         mat_t B = {1, 1, 1, 1}    // size = 1x4
         mat_t C = A.T * B;        // size = 5x4

         C = {  {0, 0, 0, 0},
                {1, 1, 1, 1},
                {2, 2, 2, 2},
                {3, 3, 3, 3},
                {4, 4, 4, 4}  }

    Аргументы:

    K - Задает размер вектора A. Эквивалентно кол-во столбцов вектора А. Кол-во строк матрицы C.

    N - Задает размер вектора В. Эквивалентно кол-во столбцов вектора В. Кол-во столбцов матрицы C.

    alpha - Множитель операции (см. определение).

    A - Задает адресс вектора А.

    B - Задает адресс вектора B.

    beta - Множитель операции (см. определение).

    C - Задает адресс матрицы C.

    ldc - Задает ведущее измерение матрицы C. Эквивалентно кол-ву столбцов вектора B.

Return Value:

    None.

--*/
#else
#error Not implemented yet
#endif

} // mmpack


#endif //XSDNN_SV2MM_H
