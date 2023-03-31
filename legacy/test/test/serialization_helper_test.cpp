//
// Created by Andrei R. on 10.01.23.
// Copyright (c) 2023 xsDNN. All rights reserved.
//

#include <gtest/gtest.h>
#include "../xsDNN.hpp"
using namespace xsdnn;

template<typename T>
void tensor_eq(const T& t_1, const T& t_2) {
    ASSERT_TRUE(t_1.size() == t_2.size());

    const Scalar* t1_data = t_1.data();
    const Scalar* t2_data = t_2.data();

    for (Index i = 0; i < t_1.size(); i++) {
        ASSERT_EQ(t1_data[i], t2_data[i]);
    }
}

TEST(Serialization, archive) {
    Tensor_3D t1(2, 2, 2); t1.setRandom();
    Tensor_3D t2(2, 2, 2); t2.setRandom();
    Tensor_3D t3(2, 2, 2); t3.setRandom();

    io::archive a(t1.size() + t2.size() + t3.size());

    a.save_wb("cerial_test", "_01", t1, t2, t3);

    Tensor_3D t11(2, 2, 2);
    Tensor_3D t22(2, 2, 2);
    Tensor_3D t33(2, 2, 2);

    a.load_wb("cerial_test", "_01", t11, t22, t33);

    tensor_eq(t1, t11);
    tensor_eq(t2, t22);
    tensor_eq(t3, t33);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}