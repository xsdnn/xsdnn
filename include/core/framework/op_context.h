//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_OP_CONTEXT_H
#define XSDNN_OP_CONTEXT_H

#include <vector>
#include "../../utils/tensor.h"
#include "../backend.h"

namespace xsdnn {
    namespace core {

class OpContext {
public:
    OpContext()
            :
            in_data_(nullptr),
            out_data_(nullptr),
            in_grad_(nullptr),
            out_grad_(nullptr),
            engine_(core::backend_t::xs),
            parallelize_(true) {}

public:
    tensor_t& input_data(const size_t index);
    const tensor_t& input_data(const size_t index) const;

    tensor_t& output_data(size_t index);
    const tensor_t& output_data(const size_t index) const;

    tensor_t& input_grad(const size_t index);
    const tensor_t& input_grad(const size_t index) const;

    tensor_t& output_grad(const size_t index);
    const tensor_t& output_grad(const size_t index) const;


    void set_in_out(const std::vector<tensor_t*>& in_data,
                    std::vector<tensor_t*>& out_data);

    void set_in_out(const std::vector<tensor_t*>& in_data,
                    const std::vector<tensor_t*>& out_data,
                    std::vector<tensor_t*>&       out_grad,
                    std::vector<tensor_t*>&       in_grad);

    void set_engine(core::backend_t engine);
    core::backend_t engine() const;

    void set_parallelize(bool parallelize);
    bool parallelize() const;

private:
    std::vector<tensor_t*>* in_data_;
    std::vector<tensor_t*>* out_data_;
    std::vector<tensor_t*>* in_grad_;
    std::vector<tensor_t*>* out_grad_;

    core::backend_t engine_;
    bool parallelize_;
};

    } // core
} // xsdnn

#endif //XSDNN_OP_CONTEXT_H
