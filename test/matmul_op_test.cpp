//
// Created by Andrei R. on 20.01.23.
// Copyright (c) 2023 xsDNN. All rights reserved.
//

#include <gtest/gtest.h>
#include "../xsDNN.hpp"
using namespace xsdnn;

TEST(matmul, vanill) {
    Matrix X(2, 2);
    X.setValues({{1, 2}, {3, 4}});
    Matrix w(2, 1);
    w.setValues({{4}, {4}});

    // result matrix
    Matrix Xw(2, 1);
    tensorize::matmul(xsThread::cpu, X, w, Xw, {DimPair(1, 0)});

    auto dim = Xw.dimensions();

    ASSERT_EQ(dim.size(), 2);
    ASSERT_EQ(dim[0], 2);
    ASSERT_EQ(dim[1], 1);

    ASSERT_EQ(Xw(0, 0), 12);
    ASSERT_EQ(Xw(1, 0), 28);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}