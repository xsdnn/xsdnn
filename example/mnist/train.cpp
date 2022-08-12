# include "../../neuralnetwork/xsDNN.h"

void build_net(NeuralNetwork& net)
{
    using FC = FullyConnected<init::Normal, activate::Identity>;
    using BN = BatchNorm1D<init::Normal, activate::LeakyReLU>;
    using DP = Dropout<activate::Identity>;

    net     << new FC(784, 784)
            << new BN(784)
            << new DP(784, 0.65)

            << new FC(784, 128)
            << new BN(128)
            << new DP(128, 0.55)

            << new FullyConnected<init::Normal, activate::Softmax>(128, 10, false);

    Output* criterion = new CrossEntropyLoss();

    net.set_output(criterion);
}


int main()
{
    system("figlet -c xsDNN");

    Matrix train_image, train_label;
    Matrix test_image,  test_label;

    dataset::parse_mnist_image("../datasets/mnist/train-images-idx3-ubyte",
                               train_image,
                               0.0,
                               1.0,
                               0.0,
                               0.0);

    dataset::parse_mnist_label("../datasets/mnist/train-labels-idx1-ubyte", train_label);

    dataset::parse_mnist_image("../datasets/mnist/t10k-images-idx3-ubyte",
                               test_image,
                               0.0,
                               1.0,
                               0.0,
                               0.0);

    dataset::parse_mnist_label("../datasets/mnist/t10k-labels-idx1-ubyte", test_label);



    NeuralNetwork fbad_baseline;

    std::vector< std::vector<Scalar> > init_params = {
            {0.0, 1.0 / std::sqrt(784.0 + 1024.0)},
            {0.0, 1.0 / std::sqrt(1024.0)},
            {},

            {0.0, 1.0 / std::sqrt(784.0 + 128.0)},
            {0.0, 1.0 / std::sqrt(128.0)},
            {},

            {0.0, 1.0 / std::sqrt(128.0 + 10.0)},
    };

    build_net(fbad_baseline);

    SGD opt; opt.m_lrate = 0.01; opt.m_decay = 0.075; opt.m_momentum = 0.75; opt.m_nesterov = true;

    fbad_baseline.fit(opt,
                      train_image,
                      train_label,
                      16,
                      5,
                      42,
                      10,
                      init_params);

    fbad_baseline.export_net("example-mnist", "fbad_baseline");
}