//
// Created by Andrei R. on 10.01.23.
// Copyright (c) 2023 xsDNN. All rights reserved.
//

#include <gtest/gtest.h>
#include "../xsDNN.hpp"
using namespace xsdnn;

TEST(Serialization, help_utils) {
    std::vector<Scalar> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3.14};
    std::string path = "serialization_helper_test_folder";
    io::make_dir(path);
    io::cerial_vector(v, path + "/" +"0");

    std::vector<Scalar> v2 = io::decerial_vector(path + "/" + "0");

    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], v2[i]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}