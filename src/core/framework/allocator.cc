//
// Created by rozhin on 26.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/framework/allocator.h>
#include <mmpack/mmpack.h>
#include <utils/xs_error.h>
#include <malloc.h>

void* DefaultCPUAlloc(size_t size) {
    constexpr size_t alignment = 64;
    if (size <= 0) return nullptr;
    void* p;
#if defined(_LIBCPP_SGX_CONFIG)
    p = memalign(alignment, size);
  if (p == nullptr)
      xsdnn::xs_error("Bad alloc.");
#else
    int ret = posix_memalign(&p, alignment, size);
    if (ret != 0)
        xsdnn::xs_error("Bad alloc.");
#endif
    return p;
}

void DefaultCPUFree(void* ptr) {
    free(ptr);
}

namespace xsdnn {

void *CPUAllocator::Alloc(size_t size) {
    return DefaultCPUAlloc(size);
}

void CPUAllocator::Free(void *ptr) {
    DefaultCPUFree(ptr);
}

}