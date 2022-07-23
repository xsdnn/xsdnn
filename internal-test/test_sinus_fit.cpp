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
# include "../neuralnetwork/DNN.h"
# include <ctime>

int main() {
    // генерируем синтетические данные
    // тренировочную выборку в диапазоне (-pi, pi)
    // тестовую выборку в диапазоне (-2pi, 2pi)
    Matrix train_data = Eigen::MatrixXd::Random(1, 200) * 3.1415f;
    Matrix train_target = train_data.array().sin();
    Matrix test_data = Eigen::MatrixXd::Random(1, 72) * 6.2832f;
    Matrix test_target = test_data.array().sin();

    // определяем объект класса нейросети - его мы будем использовать ниже
    NeuralNetwork net;

    // определяем слои сетки в формате Layer* layer_name = new Layer_type<Activation_type>(layer_params)
    Layer *l1 = new FullyConnected<activate::Sigmoid, init::Normal>(1, 10);
    Layer *l2 = new FullyConnected<activate::Sigmoid, init::Normal>(10, 10);
    Layer *l3 = new FullyConnected<activate::Identity, init::Normal>(10, 1);

    // добавляем слои в сетку - важен порядок добавления: в каком добавили, в таком и будут производится вычисления
    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);

    // определяем объект вывода информации на дисплей и устанавливаем измерение ошибки как на регрессию с критерием MSE
    VerboseCallback callback;
    net.set_callback(callback);
    net.set_output(new RegressionMSE());

    // определяем объект оптимизатора сетки и устанавливаем длину шага
    SGD opt;
    opt.m_lrate = 0.01;

    // инициализируем веса сетки со значениями параметров по умолчанию
    net.init();
    net.train();

    std::cout << "Net init - start training" << std::endl;
    // начинаем обучение сети с batch_size = 20, кол-вом эпох = 2500 и random seed = 42
    unsigned int start_time = clock();
    net.fit(opt, train_data, train_target, 20, 5000, 42);
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
    net.export_net("nn_test", "model_1");
}