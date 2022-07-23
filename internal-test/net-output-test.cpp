//
// Created by shuffle on 06.06.22.
//


# include "../neuralnetwork/DNN.h"

int main()
{
    NeuralNetwork net;

    Layer* l1 = new FullyConnected<activate::Sigmoid, init::Uniform>(1000,2000);
    Layer* l2 = new FullyConnected<activate::Sigmoid, init::Uniform>(2000,5000);
    Layer* l3 = new FullyConnected<activate::Sigmoid, init::Uniform>(5000,10000);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);

    std::cout << net;
}
