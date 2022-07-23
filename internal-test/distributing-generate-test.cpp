//
// Created by shuffle on 22.06.22.
//

# include "../neuralnetwork/DNN.h"

/*
 * Тест 1.
 *
 * Проверить общую реализацию. Подать матрицу весов для случая, когда все распределения одинаковы, и
 * для случая, когда различны.
 *
 * Test completed.
 * ----------------------------------------------------------------------------------------------------------------
 *
 * Тест 2.
 *
 * Проверить случаи, когда кол-во параметров в матрице не соответствует необходимому кол-ву для распределения
 *
 * Test completed.
 * ----------------------------------------------------------------------------------------------------------------
 *
 * Тест 3.
 *
 * В матрице описаны не все слои - для каких-то слоев нет вектора параметров распределения.
 *
 * Test completed.
 * ----------------------------------------------------------------------------------------------------------------
 */


void test_1_1()
{
    NeuralNetwork net;

    Layer* l1 = new FullyConnected<activate::ReLU, init::Uniform>(728, 512);
    Layer* l2 = new FullyConnected<activate::ReLU, init::Uniform>(512, 728);
    Layer* l3 = new FullyConnected<activate::ReLU, init::Uniform>(728, 512);
    Layer* l4 = new FullyConnected<activate::ReLU, init::Uniform>(512, 128);
    Layer* l5 = new FullyConnected<activate::ReLU, init::Uniform>(128, 64);
    Layer* l6 = new FullyConnected<activate::Identity, init::Uniform>(64, 10);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);

    std::vector<std::vector<Scalar>> distribution_params = {
            {0, 1},
            {10, 11},
            {17, 19},
            {34, 37},
            {-10, -5},
            {0, 1}
    };

    net.init(42, distribution_params);
}

void test_1_2()
{
    NeuralNetwork net;

    Layer* l1 = new FullyConnected<activate::ReLU, init::Uniform>(728, 512);
    Layer* l2 = new FullyConnected<activate::ReLU, init::Exponential>(512, 728);
    Layer* l3 = new FullyConnected<activate::ReLU, init::Normal>(728, 512);
    Layer* l4 = new FullyConnected<activate::ReLU, init::Uniform>(512, 128);
    Layer* l5 = new FullyConnected<activate::ReLU, init::Normal>(128, 64);
    Layer* l6 = new FullyConnected<activate::Identity, init::Exponential>(64, 10);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);

    std::vector<std::vector<Scalar>> distribution_params = {
            {0, 1},
            {10},
            {0.1, 100},
            {34, 37},
            {-10, 5},
            {0.1}
    };

    net.init(42, distribution_params);
}

void test_2_1()
{
    NeuralNetwork net;

    Layer* l1 = new FullyConnected<activate::ReLU, init::Uniform>(728, 512);
    Layer* l2 = new FullyConnected<activate::ReLU, init::Uniform>(512, 728);
    Layer* l3 = new FullyConnected<activate::ReLU, init::Uniform>(728, 512);
    Layer* l4 = new FullyConnected<activate::ReLU, init::Uniform>(512, 128);
    Layer* l5 = new FullyConnected<activate::ReLU, init::Uniform>(128, 64);
    Layer* l6 = new FullyConnected<activate::Identity, init::Uniform>(64, 10);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);

    std::vector<std::vector<Scalar>> distribution_params = {
            {0, 1},
            {11},
            {17, 19},
            {37},
            {-10, -5},
            {0, 1}
    };

    net.init(42, distribution_params);
}

void test_2_2()
{
    NeuralNetwork net;

    Layer* l1 = new FullyConnected<activate::ReLU, init::Uniform>(728, 512);
    Layer* l2 = new FullyConnected<activate::ReLU, init::Exponential>(512, 728);
    Layer* l3 = new FullyConnected<activate::ReLU, init::Normal>(728, 512);
    Layer* l4 = new FullyConnected<activate::ReLU, init::Uniform>(512, 128);
    Layer* l5 = new FullyConnected<activate::ReLU, init::Normal>(128, 64);
    Layer* l6 = new FullyConnected<activate::Identity, init::Exponential>(64, 10);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);

    std::vector<std::vector<Scalar>> distribution_params = {
            {0, 1},
            {10},
            {0.1, -1},
            {34, 37},
            {-10, 5},
            {0.1}
    };

    net.init(42, distribution_params);
}

void test_3()
{
    NeuralNetwork net;

    Layer* l1 = new FullyConnected<activate::ReLU, init::Uniform>(728, 512);
    Layer* l2 = new FullyConnected<activate::ReLU, init::Exponential>(512, 728);
    Layer* l3 = new FullyConnected<activate::ReLU, init::Normal>(728, 512);
    Layer* l4 = new FullyConnected<activate::ReLU, init::Uniform>(512, 128);
    Layer* l5 = new FullyConnected<activate::ReLU, init::Normal>(128, 64);
    Layer* l6 = new FullyConnected<activate::Identity, init::Exponential>(64, 10);

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);

    std::vector<std::vector<Scalar>> distribution_params = {
            {0, 1},
            {10},
            {10},
            {34, 37},
            {-10, 5},
            {0.1},
            {10}
    };

    net.init(42, distribution_params);
}
int main()
{
//    test_1_1();
//    test_1_2();
//    test_2_1();
//    test_2_2();
//    test_3();
}
