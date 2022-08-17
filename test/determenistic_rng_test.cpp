//
// Created by shuffle on 27.07.22.
//

///
/// Тестируется генерация маски Dropout слоя - при каждом forward pass, должна генерироваться уникальная маска
///

// #####################################################################
// #                         TEST COMPLETED                            #
// #####################################################################

# include "../neuralnetwork/xsDNN.h"


int main()
{
    // эти параметры передаются в метод инициализации слоя, но не используются при обучении сетки
    RNG rng(1);
    std::vector<Scalar> param = {};

    Matrix prev_layer_data(5, 1);
    prev_layer_data.setRandom();

    Dropout layer(5, 0.3);
    layer.init(param, rng);

    layer.train();

    for (int i = 0; i < 5; i++) layer.forward(prev_layer_data);
}
