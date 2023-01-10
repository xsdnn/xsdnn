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
    xsdnn::core::backend_t t = xsdnn::core::backend_t::default_cpu;
    std::cout << t;
}