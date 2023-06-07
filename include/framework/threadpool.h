//
// Created by rozhin on 07.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_THREADPOOL_H
#define XSDNN_THREADPOOL_H

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_set>

namespace xsdnn {
    namespace concurrency {

class threadpool {
public:
    explicit threadpool(size_t num_threads);
    threadpool(const threadpool&) = delete;
    threadpool operator=(const threadpool&) = delete;

    ~threadpool();

public:
    template<typename Func, typename ...Args>
    int32_t add_task(const Func& task_func, Args&&... args){
        int32_t task_idx = last_idx++;

        std::lock_guard<std::mutex> q_lock(queue_mutex);
        tasks_queue.emplace(std::async(std::launch::deferred, task_func, args...), task_idx);

        queue_cv.notify_one();
        return task_idx;
    }

    void run();

    void wait_all();

private:
    std::vector<std::thread> threads;
    std::atomic<bool> quite{ false };
    std::atomic<int64_t> last_idx{ 0 };

    std::queue<std::pair<std::future<void>, int>> tasks_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::unordered_set<int> done_ids;

    std::condition_variable completed_task_ids_cv;
    std::mutex completed_task_ids_mtx;
};

    }
}

#endif //XSDNN_THREADPOOL_H
