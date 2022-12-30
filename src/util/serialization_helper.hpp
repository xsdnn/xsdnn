//
// Created by Andrei R. on 30.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_SERIALIZATION_HELPER_HPP
#define XSDNN_SERIALIZATION_HELPER_HPP

#include <filesystem>
#include <fstream>

namespace xsdnn {
namespace io {

namespace fs = std::filesystem;

void make_dir(const std::string& __s) {
    fs::current_path("./");
    if (fs::create_directory(__s)) {
        std::cout << "Directory '" << __s << "' created successfuly" << std::endl;
    } else {
        xs_warning(" " + __s + " directory already established. You may be need to create new directory.");
    }
}

void cerial_vector(const std::vector<Scalar>& __v, std::string __s) {
    std::ofstream ofs(__s.c_str(), std::ios::out | std::ios::binary);
    if (ofs.fail()) {
        throw xs_error("[void cerial_vector] Error when opening file :: " + __s);
    }

    std::ostream_iterator<char> osi(ofs);
    const char *begin_byte = reinterpret_cast<const char *>(&__v[0]);
    const char *end_byte = begin_byte + __v.size() * sizeof(Scalar);
    std::copy(begin_byte, end_byte, osi);
}

std::vector<Scalar> decerial_vector(const std::string __s) {
    std::ifstream ifs(__s.c_str(), std::ios::in | std::ifstream::binary);
    if (ifs.fail())
        throw xs_error("[decerial_vector] Error while opening file :: " + __s);

    std::vector<char> buffer;
    std::istreambuf_iterator<char> iter(ifs);
    std::istreambuf_iterator<char> end;
    std::copy(iter, end, std::back_inserter(buffer));
    std::vector<Scalar> vec(buffer.size() / sizeof(Scalar));
    std::copy(&buffer[0], &buffer[0] + buffer.size(), reinterpret_cast<char *>(&vec[0]));
    return vec;
}

} // io
} // xsdnn

#endif //XSDNN_SERIALIZATION_HELPER_HPP
