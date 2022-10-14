//
// Created by Andrei R. on 14.10.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_TYPE_TEST_H
#define XSDNN_TYPE_TEST_H

TEST(tensor_types, unaligned) {
    Tensor_4D T4(1, 2, 3, 4); T4.setRandom(); std::cout << T4 << "\n\n\n";
    Tensor_3D T3(1, 2, 3); T3.setRandom(); std::cout << T3 << "\n\n\n";
    Matrix T2(2, 2); T2.setRandom(); std::cout << T2 << "\n\n\n";
    Vector T1(2); T1.setRandom();
}

TEST(tensor_types, aligned) {
    Tensor_4D T4(1, 2, 3, 4); T4.setRandom();
    Tensor_3D T3(1, 2, 3); T3.setRandom();
    Matrix T2(2, 2); T2.setRandom();
    Vector T1(2); T1.setRandom();

    xsTypes::AlignedTensor_4D AT4(T4.data(), T4.dimensions());
    xsTypes::AlignedTensor_3D AT3(T3.data(), T3.dimensions());
    xsTypes::AlignedMatrix AT2(T2.data(), T2.dimensions());
    xsTypes::AlignedVector AT1(T1.data(), T1.dimensions());

    Scalar* T4ptr  = T4.data();
    Scalar* AT4ptr = AT4.data();
    for (int i = 0; i < T4.size(); i++) {
        ASSERT_EQ(T4ptr[i], AT4ptr[i]);
    }

    Scalar* T3ptr  = T3.data();
    Scalar* AT3ptr = AT3.data();
    for (int i = 0; i < T3.size(); i++) {
        ASSERT_EQ(T3ptr[i], AT3ptr[i]);
    }

    Scalar* T2ptr  = T2.data();
    Scalar* AT2ptr = AT2.data();
    for (int i = 0; i < T2.size(); i++) {
        ASSERT_EQ(T2ptr[i], AT2ptr[i]);
    }

    Scalar* T1ptr  = T1.data();
    Scalar* AT1ptr = AT1.data();
    for (int i = 0; i < T1.size(); i++) {
        ASSERT_EQ(T1ptr[i], AT1ptr[i]);
    }
}
#endif //XSDNN_TYPE_TEST_H
