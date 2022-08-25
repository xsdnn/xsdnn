//
// Created by shuffle on 23.08.22.
//

#ifndef XSDNN_ACTIVATION_TEST_H
#define XSDNN_ACTIVATION_TEST_H

TEST(activation, identity_forward){
    activate::Identity a;
    Matrix Z(3, 3), A(3, 3);
    Z   <<  1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    a.activate(Z, A);
    Scalar* Z_data = Z.data();
    Scalar* A_data = A.data();
    for (int i = 0; i < Z.size(); i++){
        EXPECT_NEAR(Z_data[i], A_data[i], 1e-4);
    }
}

TEST(activation, identity_backprop){
    activate::Identity a;
    Matrix Z(3, 3), A(3, 3), F(3, 3), G(3, 3);
    A   <<  1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    F   <<  1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    Matrix expected(3, 3);
    expected    <<  1, 2, 3,
                    4, 5, 6,
                    7, 8, 9;

    a.apply_jacobian(Z, A, F, G);
    Scalar* G_data = G.data();
    Scalar* expected_data = expected.data();

    for (int i = 0; i < G.size(); i++){
        EXPECT_NEAR(G_data[i], expected_data[i], 1e-4);
    }
}

TEST(activation, LeakyReLU_forward){
    activate::LeakyReLU a;
    Matrix Z(3, 3), A(3, 3);
    Z   <<  1, -2, 3,
            4, -5, 6,
            7, -8, 9;
    a.activate(Z, A);

    Matrix expected(3, 3);
    expected    <<  1, -0.02, 3,
                    4, -0.05, 6,
                    7, -0.08, 9;
    Scalar* A_data = A.data();
    Scalar* expected_data = expected.data();
    for (int i = 0; i < A.size(); i++){
        EXPECT_NEAR(A_data[i], expected_data[i], 1e-4);
    }
}

TEST(activate, LeakyReLU_backprop){
    activate::LeakyReLU a;
    Matrix Z(3, 3), A(3, 3), F(3, 3), G(3, 3);
    A   <<  1, -0.02, 3,
            4, -0.05, 6,
            7, -0.08, 9;

    F   <<  0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9;

    Matrix expected(3, 3);
    expected    <<  0.1, -0.0002, 0.3,
                    0.4, -0.0005, 0.6,
                    0.7, -0.0008, 0.9;

    a.apply_jacobian(Z, A, F, G);
    Scalar* G_data = G.data();
    Scalar* expected_data = expected.data();

    for (int i = 0; i < G.size(); i++){
        EXPECT_NEAR(G_data[i], expected_data[i], 1e-4);
    }
}
#endif //XSDNN_ACTIVATION_TEST_H
