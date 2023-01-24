//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_NN_H
#define XSDNN_NN_H

TEST(nn, save){
    Matrix data(784, 100), image(784, 100);
    generate_sinus_data(data, image, 3.14f);

    NeuralNetwork net;
    net << new FullyConnected<init::Normal, activate::ReLU>(784, 128)
        << new FullyConnected<init::Normal, activate::Sigmoid>(128, 784);
    Output* criterion = new MSELoss();
    optim::SGD opt;

    net.set_output(criterion);
    net.fit(opt, data, image, 10, 1);
    net.export_net("export_read_test", "model");
}

TEST(nn, load){
    Matrix data(784, 100), image(784, 100);
    generate_sinus_data(data, image, 3.14f);

    NeuralNetwork net;
    net.read_net("export_read_test", "model");
}

TEST(nn, print_detail){
    NeuralNetwork net;
    using FC = FullyConnected<init::Normal, activate::ReLU>;
    using BN = BatchNorm1D<init::Normal, activate::Identity>;
    using DP = Dropout<activate::Identity>;

    net     <<  new FC(1, 100)
            <<  new FC(100, 784)
            <<  new FC(784, 2048)
            <<  new BN(2048)
            <<  new DP(2048, 0.8)
            <<  new FC(2048, 4096, false);

    std::cout << net;
}

#endif //XSDNN_NN_H
