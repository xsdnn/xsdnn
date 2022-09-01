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
    is_near_container(Z, A, epsilon);
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
    is_near_container(G, expected, epsilon);
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
    is_near_container(A, expected, epsilon);
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
    is_near_container(G, expected, epsilon);
}

TEST(activate, ReLU_forward){
    activate::ReLU a;
    Matrix Z(3, 3), A(3, 3);
    Z   <<  1, -2, 3,
            4, -5, 6,
            7, -8, 9;
    a.activate(Z, A);

    Matrix expected(3, 3);
    expected    <<  1.0, 0.0, 3.0,
                    4.0, 0.0, 6.0,
                    7.0, 0.0, 9.0;

    is_near_container(A, expected, epsilon);
}

TEST(activate, ReLU_backward){
    activate::LeakyReLU a;
    Matrix Z(3, 3), A(3, 3), F(3, 3), G(3, 3);
    A   <<  1.0, -0.02, 3.0,
            4.0, -0.05, 6.0,
            7.0, -0.08, 9.0;

    F   <<  0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9;

    Matrix expected(3, 3);
    expected    <<  0.1, 0.0, 0.3,
                    0.4, 0.0, 0.6,
                    0.7, 0.0, 0.9;

    a.apply_jacobian(Z, A, F, G);
    is_near_container(G, expected, epsilon);
}

TEST(activate, sigmoid_forward){
    activate::Sigmoid a;
    Matrix Z(3, 3), A(3, 3);
    Z   <<  1, -2, 3,
            4, -5, 6,
            7, -8, 9;
    a.activate(Z, A);

    Matrix expected(3, 3);
    expected    <<  7.31058579e-01, 1.19202922e-01, 9.52574127e-01,
                    9.82013790e-01, 6.69285092e-03, 9.97527377e-01,
                    9.99088949e-01, 3.35350130e-04, 9.99876605e-01;

    is_near_container(A, expected, epsilon);
}

TEST(activate, sigmoid_backward){
    activate::LeakyReLU a;
    Matrix Z(3, 3), A(3, 3), F(3, 3), G(3, 3);
    A    <<     7.31058579e-01, 1.19202922e-01, 9.52574127e-01,
                9.82013790e-01, 6.69285092e-03, 9.97527377e-01,
                9.99088949e-01, 3.35350130e-04, 9.99876605e-01;

    F   <<      0.1, 0.2, 0.3,
                0.4, 0.5, 0.6,
                0.7, 0.8, 0.9;

    Matrix expected(3, 3);
    expected    <<  0.02193619, 0.0498228 , 0.0602637,
                    0.07929611, 0.1249986 , 0.11810189,
                    0.13768629, 0.19999999, 0.17696083;

    a.apply_jacobian(Z, A, F, G);
    is_near_container(G, expected, epsilon);
}
#endif //XSDNN_ACTIVATION_TEST_H
