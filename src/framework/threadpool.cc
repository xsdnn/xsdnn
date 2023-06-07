//
// Created by rozhin on 07.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <framework/threadpool.h>

namespace xsdnn {
    namespace concurrency {

threadpool::threadpool(size_t num_threads) {
    threads.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(&threadpool::run, this);
    }
}

void threadpool::run() {
    while (!quite){
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [this]()->bool{return !tasks_queue.empty() || quite;});

        if (!tasks_queue.empty()){
            auto elem = std::move(tasks_queue.front());
            tasks_queue.pop();
            lock.unlock();

            try {
                elem.first.get();
            }
            catch (std::exception& exc) {
                exc.what();
            }

            std::lock_guard<std::mutex> lock_t(completed_task_ids_mtx);
            done_ids.insert(elem.second);
            completed_task_ids_cv.notify_all();
        }
    }
}

void threadpool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex);

    completed_task_ids_cv.wait(lock, [this]()->bool {
        std::lock_guard<std::mutex> task_lock(completed_task_ids_mtx);
        return tasks_queue.empty() && last_idx == done_ids.size();
    });
}

threadpool::~threadpool() {
    wait_all();
    quite = true;
    for (uint32_t i = 0; i < threads.size(); ++i) {
        queue_cv.notify_all();
        threads[i].join();
    }
}


    }
}
