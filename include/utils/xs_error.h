//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_XS_ERROR_H
#define XSDNN_XS_ERROR_H

#include <exception>
#include <string>

namespace xsdnn {

#define START_MSG                                       \
"\nFrom: " + std::string(__FILE__) + ";\n" +            \
"Function: " + std::string(__FUNCTION__) + ";\n" +      \
"Message: "

class xs_error : public std::exception {
public:
    xs_error(const std::string& msg) : msg_(msg) {}
    const char* what() const throw() override;

private:
    std::string msg_;
};

class xs_warning {
public:
    explicit xs_warning(const std::string& msg);

private:
    std::string msg_;
    std::string type_ = "[WARNING]";
};

} // xsdnn

#endif //XSDNN_XS_ERROR_H
