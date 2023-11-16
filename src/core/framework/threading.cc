//
// Created by rozhin on 16.11.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//
#include <core/framework/threading.h>

namespace xsdnn {
    namespace concurrency {
#ifdef XS_USE_XNNPACK
threadpool &threadpool::getInstance() {
    static threadpool instance;
    return instance;
}

void threadpool::create(size_t num_threads) {
    if (initialized) return;
    threadpool_ = pthreadpool_create(num_threads);
    initialized = true;
}
#endif
    }
}