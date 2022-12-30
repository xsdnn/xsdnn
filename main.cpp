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
    xsdnn::default_random_engine rng(42);
    std::cout << rng.rand();
}