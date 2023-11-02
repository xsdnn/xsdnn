//
// Created by rozhin on 10.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_INFERENCE_SESSION_H
#define XSDNN_INFERENCE_SESSION_H

#include "inference_options.h"
#include "../common/network.h"

namespace xsdnn {

class InfSession {
public:
    explicit InfSession(const InfOptions& opt);

public:
    void SetSessionOptions(const InfOptions& opt);

public:
    void Load(std::string model_path);
    void Run(const std::vector<tensor_t>& input, std::vector<tensor_t>& output);
    network GetModel();

private:
    template<typename T>
    void load_and_verify_model(T& model, std::string path) {
        if (model.empty()) {
            model.load(path);
        }
        // TODO: как можно проверить, что загрузка прошла успешно?
    }

private:
    InfOptions opt_;
    std::shared_ptr<network> net_;
};

} // xsdnn

#endif //XSDNN_INFERENCE_SESSION_H
