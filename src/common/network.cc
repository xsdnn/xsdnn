//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/network.h>

namespace xsdnn {

void network::init_weight() {
    net_.setup(true);
}

mat_t network::predict(const mat_t &in) {
    return fprop(in);
}

tensor_t network::predict(const tensor_t &in) {
    return fprop(in);
}

std::vector<tensor_t> network::predict(const std::vector<tensor_t> &in) {
    return fprop(in);
}


mat_t network::fprop(const mat_t &in) {
    return fprop(tensor_t{in})[0];
}

tensor_t network::fprop(const tensor_t &in) {
    return fprop(std::vector<tensor_t>{ in })[0];
}

std::vector<tensor_t> network::fprop(const std::vector<tensor_t> &in) {
    return net_.forward(in);
}

void network::train(loss *loss, optimizer *opt, const tensor_t &input,
                    const std::vector<size_t> &label, size_t batch_size, size_t epoch) {
    std::vector<tensor_t> input_tensor, output_tensor;
    newaxis(input, input_tensor);
    label2vec(label, output_tensor);

    fit(loss, opt, input_tensor, output_tensor, batch_size, epoch);
}

void network::fit(loss *l_ptr, optimizer *opt_ptr, std::vector<tensor_t> &input,
                  std::vector<tensor_t> &label, size_t batch_size, size_t epoch) {
    net_.setup(true);
    for (auto l : net_) {
        l->set_parallelize(true);
    }
    opt_ptr->reset();
    for (size_t e = 0; e < epoch; ++e) {
        for (size_t b = 0; b < input.size(); b += batch_size) {
            fit_batch(l_ptr,
                      opt_ptr,
                      &input[b], &label[b],
                      std::min(batch_size, input.size() - b));
        }
    }
}

void network::fit_batch(loss *l_ptr, optimizer *opt_ptr, const tensor_t *input,
                        const tensor_t *label, size_t batch_size) {
    if (batch_size == 1) {
        throw xs_error("Not implemented yet"); // FIXME: доделать
    } else if (batch_size > 1) {
        compute(l_ptr, opt_ptr, input, label, batch_size);
    } else {
        throw xs_error("Batch size <= 0");
    }
}

void network::compute(loss *l_ptr, optimizer *opt_ptr, const tensor_t *input,
                      const tensor_t *label, size_t batch_size) {
    std::vector<tensor_t> in_batch(&input[0], &input[0] + batch_size);
    std::vector<tensor_t> l_batch(&label[0], &label[0] + batch_size);
    bprop(l_ptr, opt_ptr, fprop(in_batch), l_batch);
}

void network::bprop(loss* l_ptr,
                    optimizer* opt_ptr,
                    const std::vector<tensor_t> &net_out,
                    const std::vector<tensor_t> &label) {
    std::vector<tensor_t> delta; // grad(loss)
    delta.resize(net_out.size());
    gradient(l_ptr, net_out, label, delta);
    net_.backward(delta);
    net_.update_weights(opt_ptr);
}

void network::newaxis(const tensor_t &in, std::vector<tensor_t> &out) {
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out.emplace_back(tensor_t{ in[i] });
    }
}

void network::label2vec(const std::vector<size_t> &label,
                        std::vector<tensor_t>& output) {
    size_t num_label = label.size();
    size_t outdim = net_.out_data_size();

    tensor_t predef_vec;
    predef_vec.reserve(num_label);
    for (size_t i = 0; i < num_label; ++i) {
        assert(label[i] < outdim);
        mm_scalar min_output_val = net_.nodes_.back()->out_value_range().first;
        mm_scalar max_output_val = net_.nodes_.back()->out_value_range().second;
        predef_vec.emplace_back(outdim, min_output_val);
        predef_vec[i][label[i]] = max_output_val;
    }

    newaxis(predef_vec, output);
}

} // xsdnn