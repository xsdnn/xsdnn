//
// Created by rozhin on 07.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_THREADPOOL_UPDATE_H
#define XSDNN_THREADPOOL_UPDATE_H

#include <BS_thread_pool.hpp>
#include <utils/macro.h>

namespace xsdnn {
    namespace concurrency {

class threadpool {
public:
    threadpool(size_t num_threads) : pool(num_threads) {}
    threadpool(const threadpool&) = delete;
    threadpool operator=(const threadpool&) = delete;

    ~threadpool() = default;

public:
    size_t num_threads() const { return pool.get_thread_count(); }

    template <typename F, typename... A>
    void add_task(F&& f, A&&... args) {
        pool.template push_task(f, std::forward<A>(args)...);
    }

    void wait_all() {
        pool.wait_for_tasks();
    }

private:
    BS::thread_pool pool;
};

auto available_threads = std::thread::hardware_concurrency();
static threadpool ThreadPool(
        XS_NUM_THREAD > available_threads ? available_threads : XS_NUM_THREAD
        );

    }
}

#endif //XSDNN_THREADPOOL_UPDATE_H
