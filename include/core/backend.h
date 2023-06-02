//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BACKEND_H
#define XSDNN_BACKEND_H

namespace xsdnn {
    namespace core {

enum class backend_t { xs };

backend_t default_backend_engine();

    } // core
} // xsdnn

#endif //XSDNN_BACKEND_H
