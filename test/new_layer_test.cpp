//
// Created by shuffle on 24.07.22.
//
///
///     Тестирование новых слоев
///

# include "../neuralnetwork/xsDNN.h"
# include <random>

template <typename T>
void print2d(std::vector< std::vector<T> >& obj)
{
    for (int i = 0; i < obj.size(); i++)
    {
        for (int j = 0; j < obj[i].size(); j++)
        {
            std::cout << obj[i][j] << " ";
        } std::cout << std::endl;
    }
}

template <typename T>
void print(std::vector< T >& obj)
{
    for (int i = 0; i < obj.size(); i++)
    {
        std::cout << obj[i]<< " ";
    }std::cout << std::endl;
}


int main()
{
    Matrix prev_payer_data(3, 2);

    prev_payer_data  << 0, 1,
                        2, 3,
                        4, 5;

    RNG rng(42);
    BatchNorm1D<activate::ReLU, init::Normal> layer(3);
    std::vector<Scalar> param = {0.0, 1.0};
    layer.init(param, rng);
    layer.train();

    for (int i = 0; i < 100; i++)
    {
        layer.forward(prev_payer_data);
    }

    layer.eval();
    layer.forward(prev_payer_data);
    std::cout << layer.output();
}