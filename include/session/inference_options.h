//
// Created by rozhin on 10.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_INFERENCE_OPTIONS_H
#define XSDNN_INFERENCE_OPTIONS_H

#include <cstdlib>
#include <iostream>

namespace xsdnn {

    enum class net_type {
        sequential = 0,
        graph = 1
    };

class InfOptions {
public:
    explicit InfOptions() {}

public:
    void SetNumThreads(size_t num_threads) {
        num_threads_ = num_threads;
    }

    void SetNetType(net_type type) {
        net_type_ = type;
    }

    void SetBatchSize(size_t batch_size) {
        batch_size_ = batch_size;
    }

    friend std::ostream& operator<<(std::ostream& out, const InfOptions& opt);

private:
    size_t num_threads_;
    size_t batch_size_;
    net_type net_type_;

    friend class InfSession;
};

} // xsdnn

#endif //XSDNN_INFERENCE_OPTIONS_H
