/*
 * Тестируется возможность сетки приближать функцию синуса на синтетических данных
 *
 * Результаты для 200 объектов на обучении и 5000 эпох
 *
 *   Elapsed time = 24.7389 сек.
 *   MSE Error = 0.0984116
 *   MAE Error = 0.199774
 *   R^2 = 0.811171
 *   MAPE Error = 0.0320455%
 *
 * Тест пройден.
 */
# include "../neuralnetwork/xsDNN.h"
# include <ctime>

int main() {
    // генерируем синтетические данные
    // тренировочную выборку в диапазоне (-pi, pi)
    // тестовую выборку в диапазоне (-2pi, 2pi)
    Matrix train_data = Eigen::MatrixXd::Random(1, 100) * 3.1415f;
    Matrix train_target = train_data.array().sin();
    Matrix test_data = Eigen::MatrixXd::Random(1, 72) * 6.2832f;
    Matrix test_target = test_data.array().sin();

    // определяем объект класса нейросети - его мы будем использовать ниже
    NeuralNetwork net;

    // определяем слои сетки в формате Layer* layer_name = new Layer_type<Distribution_type, Activation_type>(layer_params)
    Layer *l1 = new FullyConnected<init::Normal, activate::Sigmoid>(1, 10, false);
    Layer *l2 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l3 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l4 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l5 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l6 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l7 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l8 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l9 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l10 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l11 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l12 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l13 = new FullyConnected<init::Normal, activate::Sigmoid>(10, 10);
    Layer *l14 = new FullyConnected<init::Normal, activate::Identity>(10, 1);

    // добавляем слои в сетку - важен порядок добавления: в каком добавили, в таком и будут производится вычисления
    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);
    net.add_layer(l5);
    net.add_layer(l6);
    net.add_layer(l7);
    net.add_layer(l8);
    net.add_layer(l9);
    net.add_layer(l10);
    net.add_layer(l11);
    net.add_layer(l12);
    net.add_layer(l13);
    net.add_layer(l14);


    // Устанавливаем измерение ошибки как на регрессию с критерием MSE
    net.set_output(new MSELoss());

    // определяем объект оптимизатора сетки и устанавливаем длину шага
    SGD opt;
    opt.m_lrate = 0.01;
    opt.m_momentum = 0.73;
    opt.m_nesterov = true;

    net.train();

    std::cout << "Net init - start training" << std::endl;
    // начинаем обучение сети с batch_size = 20, кол-вом эпох = 5000 и batch_seed = 42, init_seed = 10
    unsigned int start_time = clock();
    net.fit(opt, train_data, train_target, 20, 1, 42, 10);
    unsigned int end_time = clock();

    std::cout << "Training completed, elapsed time = " << (end_time - start_time) / 1000000.0 << " sec." << std::endl;

    net.eval();
    // считаем различные метрики
    Matrix predict = net.predict(test_data);

    metrics::MSE_calculate(test_target, predict);    // 0.191714
    metrics::MAE_calculate(test_target, predict);    // 0.307732
    metrics::R_calculate(test_target, predict);      // 0.61454
    metrics::MAPE_calculate(test_target, predict);   // 0.686659 %

    // можно так
    //regressor_metrics_calculate(test_target, predict);

    // сохраняем сетку для того, чтобы не обучать ее заново
    // folder_name, model_name
    net.export_net("nn_test", "leaky_relu_test");
    net.read_net("nn_test", "leaky_relu_test");

    std::cout << net;
}