//
// Created by shuffle on 06.06.22.
//


# include "../neuralnetwork/xsDNN.h"

int main()
{
    Matrix test_data(784, 64); test_data.setRandom() * 6.28f;
    Matrix test_label = test_data.array().sin().cos().tan();

    NeuralNetwork net;

    net.read_net("batchnorm1d_layer_test", "large_fbad_64_batch_model");

    net.eval();
    Matrix predict = net.predict(test_data);

    metrics::regressor_metrics_calculate(test_label, predict);
}
