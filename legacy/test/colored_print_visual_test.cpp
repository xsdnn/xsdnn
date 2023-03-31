//
// Created by Andrei R. on 10.01.23.
// Copyright (c) 2023 xsDNN. All rights reserved.
//

#include <gtest/gtest.h>
#include "../xsDNN.hpp"
using namespace xsdnn;


TEST(ColoredPrint, _01) {
    std::cout << cc::red << "NotImplementedError" << std::endl;
    std::cout << "This text must be normal!" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
