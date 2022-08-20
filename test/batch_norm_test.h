//
// Created by shuffle on 18.08.22.
//

#ifndef XSDNN_BATCH_NORM_TEST_H
#define XSDNN_BATCH_NORM_TEST_H

# include "../xsDNN/Utils/Math.h"

TEST(batchnorm, init){
    Layer* bn_layer = new BatchNorm1D<init::Normal, activate::Identity>(10);
    std::vector<Scalar> init_params = {0.0, 1.0 / 10.0};
    RNG rng(1);

    bn_layer->init(init_params, rng);
    std::vector<Scalar> bn_layer_params = bn_layer->get_parametrs();
    EXPECT_EQ(bn_layer_params.size(), 10 + 10 + 10 + 10);
}

TEST(batchnorm, forward){
    BatchNorm1D bn_layer = BatchNorm1D<init::Constant, activate::Identity>(3, false);
    std::vector<Scalar> init_params = {0.0};
    RNG rng(1); bn_layer.init(init_params, rng);

    Matrix input_data(3, 3);
    input_data  <<  1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0;

    Matrix expected(3, 3);
    expected    <<  -1.22473569,  0.0,  1.22473569,
                    -1.22473569,  0.0,  1.22473569,
                    -1.22473569,  0.0,  1.22473569;

    bn_layer.train();
    bn_layer.forward(input_data);

    Matrix output = bn_layer.output();

    Scalar* expected_data = expected.data();
    Scalar* output_data   = output.data();

    for (int i = 0; i < input_data.size(); i++){
        EXPECT_NEAR(output_data[i], expected_data[i], 1e-4);
    }
}

TEST(batchnorm, forward_affine){
    typedef Eigen::VectorXd Vector;
    Vector gamma(3), beta(3); gamma.setOnes(); beta.setZero();

    BatchNorm1D bn_layer = BatchNorm1D<init::Constant, activate::Identity>(3, true);
    std::vector<Scalar> init_params = {0.0};
    RNG rng(1); bn_layer.init(init_params, rng);
    bn_layer.set_gamma(gamma); bn_layer.set_beta(beta);

    Matrix input_data(3, 3);
    input_data  <<  1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0;

    Matrix expected(3, 3);
    expected    <<  -1.22473569,  0.0,  1.22473569,
                    -1.22473569,  0.0,  1.22473569,
                    -1.22473569,  0.0,  1.22473569;

    bn_layer.train();
    bn_layer.forward(input_data);

    Matrix output = bn_layer.output();

    Scalar* expected_data = expected.data();
    Scalar* output_data   = output.data();

    for (int i = 0; i < input_data.size(); i++){
        EXPECT_NEAR(output_data[i], expected_data[i], 1e-4);
    }
}

#endif //XSDNN_BATCH_NORM_TEST_H
