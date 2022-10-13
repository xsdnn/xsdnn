//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

# include "../../xsDNN/xsDNN.h"

int main() {
    Matrix train_data(2, 2048), test_data(2, 1024);
    train_data.setRandom(); test_data.setRandom();

    Matrix train_label = train_data.array().sin();
    Matrix test_label = test_data.array().sin();

    NeuralNetwork nn;

    nn  << new FullyConnected<init::Normal, activate::Sigmoid>(2, 10)
        << new FullyConnected<init::Normal, activate::Identity>(10, 2);

    Output* criterion = new MSELoss();
    nn.set_output(criterion);

    optim::SGDAdaptive opt(
            128,
            1,
            0.1,
            0.5
            );

    nn.fit(opt, train_data, train_label, 16, 50, 1);

    Matrix predict = nn.predict(test_data);

    std::cout << "Mean MSE = " << (predict - test_label).squaredNorm() / 1024 << std::endl;
}
