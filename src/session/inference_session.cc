//
// Created by rozhin on 10.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <session/inference_session.h>

namespace xsdnn {

    InfSession::InfSession(const xsdnn::InfOptions &opt) : opt_(opt) {}

    void InfSession::Load(std::string model_path) {
        net_.reset(new network);
        net_->load(model_path); // TODO: Verify this
    }

    void InfSession::Run(const std::vector<tensor_t> &input,
                         std::vector<tensor_t> &output) {
        assert(input.size() == net_->net_.input_layers_.size());
        assert(output.size() == net_->net_.output_layers_.size());
        output = net_->predict(input);
    }

    network InfSession::GetModel() {
        return *net_.get();
    }

} // xsdnn