//
// Created by shuffle on 03.08.22.
//

# include "../neuralnetwork/xsDNN.h"

int main()
{
    Matrix train_data(784, 1024); train_data.setRandom() * 3.14f;
    Matrix test_data(784, 512); test_data.setRandom() * 6.28f;

    Matrix train_label = train_data.array().sin().cos().tan();
    Matrix test_label = test_data.array().sin().cos().tan();

    std::vector < std::vector<Scalar> > init_params = {
            {0.0, 1.0 / (784.0 + 1024.0)},
            {},
            {0.0, 1.0 / (1024.0 + 1024.0)},
            {},
            {0.0, 1.0 / (784.0 + 1024.0)},
            {},
            {0.0, 1.0 / (784.0 + 784.0)}
    };

    NeuralNetwork net;

    Layer* l1 = new FullyConnected<init::Normal, activate::LeakyReLU>(784, 1024, true);
    Layer* l2 = new Dropout<activate::Identity>(1024, 0.81);
    Layer* l3 = new FullyConnected<init::Normal, activate::LeakyReLU>(1024, 1024, false);
    Layer* l4 = new Dropout<activate::Identity>(1024, 0.75);
    Layer* l5 = new FullyConnected<init::Normal, activate::LeakyReLU>(1024, 784, true);
    Layer* l6 = new Dropout<activate::Identity>(784, 0.85);
    Layer* l7 = new FullyConnected<init::Normal, activate::Identity>(784, 784, false);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);
    net.add_layer(l7);

    SGD opt; opt.m_lrate = 0.1; opt.m_momentum = 0.95; opt.m_nesterov = false;

    net.set_output(new MSELoss());

    net.train();
    net.fit(opt, train_data, train_label, 16, 5, 42, 10, init_params);
    net.eval();

    Matrix predict = net.predict(test_data);

    metrics::regressor_metrics_calculate(test_label, predict);

    net.export_net("dropout_layer_test", "with_large_dropout_leaky_relu");
}