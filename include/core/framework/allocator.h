//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_ALLOCATOR_H
#define XSDNN_ALLOCATOR_H

#include <cstdlib>

namespace xsdnn {

class IAllocator {
public:
    virtual ~IAllocator() = default;
    virtual void* Alloc(size_t size) = 0;
    virtual void Free(void* ptr) = 0;
};

class CPUAllocator : public IAllocator {
public:
    void* Alloc(size_t size);
    void Free(void* ptr);
};

} // xsdnn

#endif //XSDNN_ALLOCATOR_H
