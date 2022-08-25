//
// Created by shuffle on 18.08.22.
//

#ifndef XSDNN_BATCH_NORM_TEST_H
#define XSDNN_BATCH_NORM_TEST_H

# include "../xsDNN/Utils/Math.h"

TEST(batchnorm1d, init){
    Layer* bn_layer = new BatchNorm1D<init::Normal, activate::Identity>(10);
    std::vector<Scalar> init_params = {0.0, 1.0 / 10.0};
    RNG rng(1);

    bn_layer->init(init_params, rng);
    std::vector<Scalar> bn_layer_params = bn_layer->get_parametrs();
    EXPECT_EQ(bn_layer_params.size(), 10 + 10 + 10 + 10);
    Matrix test = bn_layer->backprop_data();
    std::cout << test;
}

TEST(batchnorm1d, forward){
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

    SGD opt;
    bn_layer.update(opt);

    output = bn_layer.output();
    output_data   = output.data();

    for (int i = 0; i < input_data.size(); i++){
        EXPECT_NEAR(output_data[i], expected_data[i], 1e-4);
    }
}

TEST(batchnorm1d, forward_affine){
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

    SGD opt;
    bn_layer.update(opt);

    output = bn_layer.output();
    output_data   = output.data();

    for (int i = 0; i < input_data.size(); i++){
        EXPECT_NEAR(output_data[i], expected_data[i], 1e-4);
    }
}

TEST(batchnorm1d, grad){
    const int nsamples  = 16;
    const int dim       = 2;
    BatchNorm1D bn_layer = BatchNorm1D<init::Constant, activate::Identity>(dim, false);
    std::vector<Scalar> init_params = {0.0};
    RNG rng(1); bn_layer.init(init_params, rng);

    Scalar top_diff[] = {
            0.554228544,
            -0.823364496,
            -0.103415221,
            0.669684947,
            0.142640188,
            -0.171076611,
            0.292261183,
            -0.067076027,
            -0.00277741,
            0.058186941,
            0.046050139,
            -0.006042562,
            -0.004771964,
            0.025202896,
            -0.062344212,
            0.030099955,
            -0.023314178,
            -0.030725746,
            0.070954606,
            0.055909708,
            -0.019887319,
            0.076775789,
            0.014769247,
            -0.025637595,
            0.004412052,
            -0.013895055,
            -0.001271803,
            3.15E-05,
            -0.013110356,
            0.008091689,
            -0.005485342,
            0.007250476
    };


    Scalar top_data[] = {
            -0.430924207,
            -2.23937607,
            1.7876749,
            1.41079676,
            0.578419685,
            0.662835836,
            -2.1911881,
            -0.002405337,
            1.49315703,
            -0.836038888,
            0.006807627,
            -0.012308626,
            0.424309582,
            -0.56077528,
            0.095194906,
            0.34416762,
            -0.755284429,
            -1.02720368,
            0.802836478,
            -0.06101859,
            2.17714667,
            -0.994640052,
            -0.497716337,
            0.397495717,
            -0.545207798,
            0.320612997,
            -0.016919944,
            0.102396645,
            0.551594019,
            -1.44724381,
            -0.530790627,
            0.993595243
    };

    Scalar stddev[] = {
            5.13347721,
            6.15658283
    };

    Scalar expected_gradients[] = {
            0.115063809,
            -0.100263298,
            -0.078099057,
            0.083551511,
            0.025724376,
            -0.024521608,
            0.026721604,
            -0.01322682,
            -0.049858954,
            0.030313857,
            0.003235561,
            -0.006351554,
            0.000483762,
            -0.002936667,
            -0.011636967,
            0.00547356,
            0.012069567,
            0.018599048,
            -0.015254314,
            0.007145011,
            0.01277818,
            0.001789367,
            -0.004100761,
            -0.003131026,
            0.011310737,
            -0.017643189,
            -0.005286998,
            -0.008531732,
            0.000200434,
            -0.013175356,
            -0.007668978,
            0.007226899
    };

    Matrix prev_layer_data(dim, nsamples),
    m_z(dim, nsamples),
    next_layer_backprop(dim, nsamples),
    expected_gradient(dim, nsamples);

    prev_layer_data.setZero();
    int index = 0;
    for (int i = 0; i < 4; i++){
        for (int j = i * 8; j < i * 8 + 4; j++){
            m_z.col(index) << top_data[j], top_data[j + 4];
            next_layer_backprop.col(index) << top_diff[j], top_diff[j + 4];
            expected_gradient.col(index++) << expected_gradients[j], expected_gradients[j + 4];
        }
    }
    internal::math::Vector stddev_(2); stddev_ << stddev[0], stddev[1];

    bn_layer.set_m_z(m_z);
    bn_layer.set_stddev(stddev_);
    bn_layer.backprop(prev_layer_data, next_layer_backprop);

    Scalar* expected = expected_gradient.data();
    Scalar* real     = const_cast<Scalar*>(bn_layer.backprop_data().data());

    for (int i = 0; i < expected_gradient.size(); i++){
        EXPECT_NEAR(expected[i], real[i], 1e-8);
    }
}

// TODO: написать проверку сохранения batchnorm слоя

TEST(batchnorm1d, fully_net){
    using BN = BatchNorm1D<init::Normal, activate::Identity>;
    using FC = FullyConnected<init::Normal, activate::LeakyReLU>;

    Matrix train_data(784, 160), train_label(10, 160);
    generate_sinus_data(train_data, train_label, 3.14f);

    NeuralNetwork net;
    net     <<  new FC(784, 128)
            <<  new BN(128)
            <<  new FC(128, 784);
    Output* criterion = new MSELoss();
    net.set_output(criterion);

    SGD opt; opt.m_lrate = 0.1;
    net.fit(opt, train_data, train_label, 16, 5, 10);
}

#endif //XSDNN_BATCH_NORM_TEST_H
