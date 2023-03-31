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

class archive {
public:
    archive() = default;
    archive(const size_t size) : wb(size) {}

    template<typename... Args>
    void save_wb(const std::string& path, const std::string& filename, Args... args) {
        if (wb.size() == 0) {
#ifdef XS_USE_LOG
            std::string msg = "You not init the wb vector. You make programming mistake!";
            logger::log(logger::type::error, msg);
#endif
            xs_error("Use ctor with 'size' argument when you try to cerial wb!");
        }
        assert(curr_position == 0);
        copy_tensor_to_vector(args...);
        make_dir(path);
        cerial_vector(wb, path + "/" + filename);
        assert(curr_position == wb.size());
        this->reset();
    }

    template<typename... Args>
    void load_wb(const std::string& path, const std::string& filename, Args&... args) {
        wb = this->decerial_vector(path + "/" + filename);
        assert(curr_position == 0);
        copy_vector_to_tensor(args...);
        assert(curr_position == wb.size());
        this->reset();
    }

private:
    std::vector<Scalar> wb; // weights && biases
    size_t curr_position = 0;

private:
    template<typename TensorType, typename... Args>
    void copy_tensor_to_vector(const TensorType& tensor, const Args&... args) {
        copy_tensor_to_vector(tensor);
        copy_tensor_to_vector(args...);
    }

    template<typename TensorType>
    void copy_tensor_to_vector(const TensorType& tensor) {
        std::copy(tensor.data(), tensor.data() + tensor.size(), wb.begin() + curr_position);
        curr_position += tensor.size();
    }

    template<typename TensorType, typename... Args>
    void copy_vector_to_tensor(TensorType& tensor, Args&... args) {
        copy_vector_to_tensor(tensor);
        copy_vector_to_tensor(args...);
    }

    template<typename TensorType>
    void copy_vector_to_tensor(TensorType& tensor) {
        std::copy(wb.begin() + curr_position, wb.begin() + curr_position + tensor.size(), tensor.data());
        curr_position += tensor.size();
    }

    void make_dir(const std::string& __s) {
        fs::current_path("./");
        if (fs::create_directory(__s)) {
#ifdef XS_USE_LOG
            std::string msg = "Directory '" + __s + "' created sucessfully.";
            logger::log(logger::type::info, msg);
#endif
        } else {
#ifdef XS_USE_LOG
            std::string msg = "Directory '" + __s + "' already established. You may be need to create new directory.";
            logger::log(logger::type::warn, msg);
#endif
        }
    }

    /*
     * Save vector to binary output format
     */
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

    /*
     * Load vector from binary output format
     */
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

    void reset() {
        curr_position = 0;
        wb.clear();
    }
};

} // io
} // xsdnn

#endif //XSDNN_SERIALIZATION_HELPER_HPP
