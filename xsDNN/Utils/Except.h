//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_EXCEPT_H
#define XSDNN_EXCEPT_H

#include <stdexcept>
#include <iostream>

namespace xsdnn {
    namespace internal {
        namespace except {
            class xs_error : public std::exception {
            public:
                explicit xs_error(const std::string &msg) : msg_(msg) {}

                const char *what() const throw() override { return msg_.c_str(); }

            private:
                std::string msg_;
            };

            class xs_warn {
            public:
                explicit xs_warn(std::string &message) : msg_(message) {
                    std::string cout_mess = "\x1B[93m";
                    cout_mess += msg_w;
                    cout_mess += msg_;
                    cout_mess += "\033[0m";
                    std::cout << cout_mess << std::endl;
                }

            private:
                std::string msg_;
                std::string msg_w = "[WARNING] ";
            };
        } // namespace except
    } // namespace internal
} // namespace xsdnn


#endif //XSDNN_EXCEPT_H
