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
    std::cout << xsdnn::uniform_rand<xsdnn::Scalar>(0, 10);
}