#pragma once



namespace mmpack {

#ifdef MM_USE_DOUBLE
typedef double mm_scalar;
#else
typedef float mm_scalar;
#endif

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
typedef enum { CblasNoTrans=111, CblasTrans=112 } CBLAS_TRANSPOSE;
#endif

} // mmpack
