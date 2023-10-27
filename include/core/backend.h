//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BACKEND_H
#define XSDNN_BACKEND_H

#ifdef XS_USE_XNNPACK
#include <xnnpack.h>
#endif

namespace xsdnn {
    namespace core {

enum class backend_t { xs, xnnpack };

backend_t default_backend_engine();

#ifdef XS_USE_XNNPACK
// Поддерживаемые операторы
enum class __xnn_operator {
    conv_nchw_fp32
};


// Singletone pattern
class XNNCompiler {
public:
    XNNCompiler() = default;
    XNNCompiler(const XNNCompiler&) = delete;
    XNNCompiler(const XNNCompiler&&) = delete;
    XNNCompiler& operator=(const XNNCompiler&) = delete;
    XNNCompiler& operator=(XNNCompiler&&) = delete;

public:
    static XNNCompiler& getInstance();
    void initialize();

private:
    bool initialized_{false};
};
#endif

    } // core
} // xsdnn

#endif //XSDNN_BACKEND_H
