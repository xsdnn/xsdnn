//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "backend.h"

namespace xsdnn {
    namespace core {

inline backend_t default_backend_engine() {
    return backend_t::xs;
}


    } // core
} // xsdnn