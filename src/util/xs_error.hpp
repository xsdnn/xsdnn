//
// Created by Andrei R. on 29.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_XS_ERROR_HPP
#define XSDNN_XS_ERROR_HPP

#include <exception>

namespace xsdnn {

class xs_error : std::exception {
public:
    xs_error(const std::string& msg) : msg_(msg) {}
    const char* what() const throw() override {return msg_.c_str();}

private:
    std::string msg_;
};

class xs_warning {
public:
    explicit xs_warning(const std::string& msg) : msg_(msg) {
        std::cout << cc::red << type_ + msg_ << std::endl;
    }

private:
    std::string msg_;
    std::string type_ = "[WARNING]";
};

} // xsdnn


#endif //XSDNN_XS_ERROR_HPP
