//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_TENSOR_H
#define XSDNN_TENSOR_H

#include <mmpack/mmpack.h>
#include <pector/malloc_allocator.h>

using namespace mmpack;

namespace xsdnn {
    typedef mmpack::vector<mm_scalar,
            pt::malloc_allocator<mm_scalar, true, false>,
            size_t> vec_t;

    typedef std::vector<mm_scalar, aligned_allocator<mm_scalar, 64>> mat_t;
    typedef std::vector<mat_t> tensor_t;
}


#endif //XSDNN_TENSOR_H
