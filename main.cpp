//
// Created by Andrei R. on 30.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "xsDNN.hpp"
using json = nlohmann::json;

int main() {
    xsdnn::Tensor_3D t(3, 4, 4); // in = 10, out = 20
    xsdnn::xavier w;
    w.fill(t.data(), t.size(), 16, 32);
    std::cout << t.format(Eigen::TensorIOFormat::Numpy()) << std::endl;
}