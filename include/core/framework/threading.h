//
// Created by rozhin on 12.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_THREADING_H
#define XSDNN_THREADING_H

#include <cstddef>
#include <cassert>
#include <future>
#include <vector>

namespace xsdnn {
namespace detail {

class range {
public:
    range(size_t begin, size_t end) : begin_(begin), end_(end) {}

public:
    size_t begin() { return begin_; }
    size_t end() { return end_; }

    const size_t begin() const { return begin_; }
    const size_t end() const { return end_; }

private:
    size_t begin_;
    size_t end_;
};

template<typename T, typename Func>
void SingleThreadParallelFor(T start, T end, const Func& f) {
    range r(start, end);
    f(r);
}

template<typename T, typename Func>
void MultiThreadParallelFor(size_t num_threads, T start, T end, const Func& f) {
    assert(end >= start);
    size_t blockSize = (end - start) / num_threads;
    if (blockSize * num_threads < end - start) blockSize++;

    size_t blockBegin = start;
    size_t blockEnd = blockBegin + blockSize;
    if (blockEnd > end) blockEnd = end;

    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        futures.push_back(
                std::move(std::async(std::launch::async, [blockBegin, blockEnd, &f] {
                    f(range(blockBegin, blockEnd));
                }))
                );

        blockBegin += blockSize;
        blockEnd = blockBegin + blockSize;
        if (blockBegin >= end) break;
        if (blockEnd > end) blockEnd = end;
    }

    for (auto& future : futures) future.wait();
}

template<typename T, typename Func>
void TryParallelFor(bool parallel, size_t num_threads, T start, T end, const Func& f) {
    if (parallel) {
        if (num_threads < 1) {
            SingleThreadParallelFor(start, end, f);
        } else {
            MultiThreadParallelFor(num_threads, start, end, f);
        }
    } else {
        SingleThreadParallelFor(start, end, f);
    }
}

} // detail



namespace concurrency {

template<typename T, typename Func>
void TryParallelFor(bool parallel, size_t num_threads, T end, Func f) {

    detail::TryParallelFor(parallel, num_threads, 0u, end, [&](const detail::range& r) {
        for(size_t i = r.begin(); i < r.end(); ++i) {
            f(i);
        }
    });

}

} // concurrency
} // xsdnn

#endif //XSDNN_THREADING_H
