//
// Created by shuffle on 09.08.22.
//

# include "../neuralnetwork/xsDNN.h"

int main()
{
    system("figlet -c xsDNN");

    Matrix train_data(784, 1024); train_data.setRandom() * 3.14f;
    Matrix test_data(784, 512); test_data.setRandom() * 6.28f;

    Matrix train_label = train_data.array().sin().cos().tan();
    Matrix test_label = test_data.array().sin().cos().tan();

    std::vector < std::vector<Scalar> > init_params = {
            {0.0, 1.0 / (784.0 + 1024.0)},
            {0.0, 1.0 / (1024.0 + 1024.0)},
            {},

            {0.0, 1.0 / (1024.0 + 1024.0)},
            {0.0, 1.0 / (1024.0 + 1024.0)},
            {},

            {0.0, 1.0 / (1024.0 + 1024.0)},
            {0.0, 1.0 / (1024.0 + 1024.0)},
            {},

            {0.0, 1.0 / (1024.0 + 784.0)},
            {0.0, 1.0 / (784.0 + 784.0)},
            {},

            {0.0, 1.0 / (784.0 + 784.0)}
    };

    NeuralNetwork net;

    Layer* l1 = new FullyConnected<init::Normal, activate::Identity>(784, 1024, true);
    Layer* l2 = new BatchNorm1D<init::Normal, activate::LeakyReLU>(1024);
    Layer* l3 = new Dropout<activate::Identity>(1024, 0.65);

    Layer* l4 = new FullyConnected<init::Normal, activate::Identity>(1024, 1024, true);
    Layer* l5 = new BatchNorm1D<init::Normal, activate::LeakyReLU>(1024);
    Layer* l6 = new Dropout<activate::Identity>(1024, 0.55);

    Layer* l7 = new FullyConnected<init::Normal, activate::Identity>(1024, 1024, true);
    Layer* l8 = new BatchNorm1D<init::Normal, activate::LeakyReLU>(1024);
    Layer* l9 = new Dropout<activate::Identity>(1024, 0.45);

    Layer* l10 = new FullyConnected<init::Normal, activate::Identity>(1024, 784, true);
    Layer* l11 = new BatchNorm1D<init::Normal, activate::LeakyReLU>(784);
    Layer* l12 = new Dropout<activate::Identity>(784, 0.65);

    Layer* l13 = new FullyConnected<init::Normal, activate::LeakyReLU>(784, 784, true);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);
    net.add_layer(l7);
    net.add_layer(l8);
    net.add_layer(l9);
    net.add_layer(l10);
    net.add_layer(l11);
    net.add_layer(l12);
    net.add_layer(l13);

    SGD opt; opt.m_lrate = 0.001; opt.m_momentum = 0.75; opt.m_nesterov = true;

    net.set_output(new MSELoss());

    net.train();
    net.fit(opt, train_data, train_label, 64, 5, 42, 10, init_params);
    net.eval();

    Matrix predict = net.predict(test_data);

    metrics::regressor_metrics_calculate(test_label, predict);

    net.export_net("batchnorm1d_layer_test", "large_fbad_64_batch_model");
}


