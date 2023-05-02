//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xs_error.h"
#include "color_print.h"

namespace xsdnn {
    const char* xs_error::what() const throw() {
        return msg_.c_str();
    }

    xs_warning::xs_warning(const std::string &msg) {
        msg_ = msg;
        std::cout << cc::red << type_ + msg_ << std::endl;
    }
} // xsdnn