# include "../../neuralnetwork/xsDNN.h"

void calculate_mseloss(Matrix& label, Matrix& predict)
{
    const long ncols = predict.cols();
    Matrix model_answer(1, ncols);
    Matrix relabel(1, ncols);

    for (int i = 0; i < ncols; i++)
    {
        Scalar max_elem = predict.col(i).maxCoeff();

        for (int j = 0; j < 10; j++)
        {
            if (predict(j, i) == max_elem)
            {
                model_answer(0, i) = j;
                break;
            }
        }
    }

    for (int i = 0; i < ncols; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (label(j, i) == 1)
            {
                relabel(0, i) = j;
                break;
            }
        }
    }

    Scalar error = (relabel - model_answer).squaredNorm() / ncols;
    std::cout << "RMSE Error = " << std::sqrt(error) << std::endl;
}

int main()
{
    Matrix test_image, test_label;
    dataset::parse_mnist_image("../datasets/mnist/t10k-images-idx3-ubyte",
                               test_image,
                               0.0,
                               1.0,
                               0.0,
                               0.0);
    dataset::parse_mnist_label("../datasets/mnist/t10k-labels-idx1-ubyte", test_label);

    NeuralNetwork net;
    net.read_net("example-mnist", "baseline");

    net.eval();
    Matrix predict = net.predict(test_image);

    calculate_mseloss(test_label, predict);
}