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

} // mmpack


#endif //XSDNN_MMPACK__H
