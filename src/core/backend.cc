//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <core/backend.h>
#include <utils/xs_error.h>

namespace xsdnn {
    namespace core {

backend_t default_backend_engine() {
    return backend_t::xs;
}

XNNCompiler &XNNCompiler::getInstance() {
    static XNNCompiler instance;
    return instance;
}

void XNNCompiler::initialize() {
    if (initialized_) return;

    xnn_status status = xnn_initialize(nullptr);

    if (status != xnn_status_success) {
        throw xs_error("Error when initializing XNNPACK");
    }

    initialized_ = true;
}

    } // core
} // xsdnn