//
// Created by shuffle on 10.08.22.
//

# include "../neuralnetwork/xsDNN.h"

void baseline(Matrix& train_data,
              Matrix& train_label,
              Matrix& test_data,
              Matrix& test_label)
{
    NeuralNetwork net;

    Layer* l1 = new FullyConnected<init::Normal, activate::ReLU>(784, 128, true);
    Layer* l2 = new FullyConnected<init::Normal, activate::Softmax>(128, 10, true);

    std::vector< std::vector<Scalar> > init_params = {
            {0.0, 1.0 / (784.0 + 128.0)},
            {0.0, 1.0 / (128.0 + 10.0)}
    };

    net.add_layer(l1);
    net.add_layer(l2);

    Output* criterion = new CrossEntropyLoss();
    net.set_output(criterion);

    SGD opt; opt.m_lrate = 0.1;

    net.train();
    net.fit(opt, train_data, train_label, 16, 5, 42, 10, init_params);

    net.export_net("mnist_experiment", "scratch_baseline");
}

int main(){
    system("figlet -c xsDNN");

    Matrix train_data, train_label;
    Matrix test_data,  test_label;

    dataset::parse_mnist_label("../internal-test/mnist/train-labels-idx1-ubyte", train_label);
    dataset::parse_mnist_image("../internal-test/mnist/train-images-idx3-ubyte",
                               train_data,
                               0,
                               255,
                               0,
                               0);

    dataset::parse_mnist_label("../internal-test/mnist/t10k-labels-idx1-ubyte", test_label);
    dataset::parse_mnist_image("../internal-test/mnist/t10k-images-idx3-ubyte", test_data,
                               0,
                               255,
                               0,
                               0);

    baseline(train_data, train_label, test_data, test_label);
}