//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_OP_CONTEXT_H
#define XSDNN_OP_CONTEXT_H

#include <vector>
#include "tensor.h"
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
            parallelize_(true),
            num_threads_(0) {}

public:
    BTensor& input_data(const size_t index);
    const BTensor& input_data(const size_t index) const;

    BTensor& output_data(size_t index);
    const BTensor& output_data(const size_t index) const;

    BTensor& input_grad(const size_t index);
    const BTensor& input_grad(const size_t index) const;

    BTensor& output_grad(const size_t index);
    const BTensor& output_grad(const size_t index) const;


    void set_in_out(const std::vector<BTensor*>& in_data,
                    std::vector<BTensor*>& out_data);

    void set_in_out(const std::vector<BTensor*>& in_data,
                    const std::vector<BTensor*>& out_data,
                    std::vector<BTensor*>&       out_grad,
                    std::vector<BTensor*>&       in_grad);

    void set_engine(core::backend_t engine);
    core::backend_t engine() const;

    void set_parallelize(bool parallelize);
    void set_num_threads(size_t num_threads);

    bool parallelize() const;
    size_t num_threads() const;

private:
    std::vector<BTensor*>* in_data_;
    std::vector<BTensor*>* out_data_;
    std::vector<BTensor*>* in_grad_;
    std::vector<BTensor*>* out_grad_;

    core::backend_t engine_;
    bool parallelize_;
    size_t num_threads_;
};

    } // core
} // xsdnn

#endif //XSDNN_OP_CONTEXT_H
