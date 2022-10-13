# include "../../xsDNN/xsDNN.h"
using namespace xsdnn;

void calculate_accuracy(const Matrix& target, const Matrix& predict) {
    const int nobj = target.cols();
    auto argmax_vec = internal::math::colargmax_vector(predict);
    Scalar nom = 0;
    for (int i = 0; i < nobj; i++) {
        for (int j = 0; j < 10; j++) {
            if (target(j, i) == Scalar(1)) {
                if (j == argmax_vec[i]) nom++;
            }
        }
    }
    std::cout << "Accuracy = " << nom / nobj * 100 << "%" << std::endl;
}

int main()
{
    Matrix test_image, test_label;
    dataset::parse_mnist_image("../../datasets/mnist/t10k-images-idx3-ubyte",
                               test_image,
                               0.0,
                               1.0,
                               0.0,
                               0.0);
    dataset::parse_mnist_label("../../datasets/mnist/t10k-labels-idx1-ubyte", test_label);

    NeuralNetwork net;
    net.read_net("example_old-mnist", "uniform_dist");

    net.eval();
    Matrix predict = net.predict(test_image);

    calculate_accuracy(test_label, predict);
}