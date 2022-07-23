//
// Created by shuffle on 22.06.22.
//

/*
 * Тестируется сохранение и загрузка сети.
 *
 * Test completed.
------------------------------------------------------------------------------------------------------------------------

    Metrics in train data before saving...
    MSE Error = 142277
    MAE Error = 76.3674
    R^2 = -0.000322721
    MAPE Error = 0.158914%
    Metrics in test data before saving...
    MSE Error = 14151.7
    MAE Error = 56.6268
    R^2 = -0.0110701
    MAPE Error = 0.442658%


------------------------------------------------------------------------------------------------------------------------

    MSE Error = 142294
    MAE Error = 97.1879
    R^2 = -0.000447185
    MAPE Error = 0.158914%
    Metrics in test data after saving...
    MSE Error = 14060.3
    MAE Error = 63.5321
    R^2 = -0.00454416
    MAPE Error = 0.442658%

 */

# include <iostream>
# include "../neuralnetwork/DNN.h"

int main()
{
    Eigen::MatrixXd train_data = Eigen::MatrixXd::Random(10, 728) * 3.1415f;
    Eigen::MatrixXd train_label = train_data.array().tan();
    Eigen::MatrixXd test_data = Eigen::MatrixXd::Random(10, 256) * 6.283f;
    Eigen::MatrixXd test_label = test_data.array().tan();

    NeuralNetwork net;

    Layer* l1 = new FullyConnected<activate::Sigmoid, init::Normal>(10, 350);
    Layer* l2 = new FullyConnected<activate::ReLU, init::Normal>(350, 250);
    Layer* l3 = new FullyConnected<activate::Sigmoid, init::Normal>(250, 100);
    Layer* l4 = new FullyConnected<activate::Identity, init::Uniform>(100, 10);

    net.set_output(new RegressionMSE());

    VerboseCallback call;
    net.set_callback(call);

    std::vector< std::vector<Scalar> > distribution_params = {
            {0.0, 0.01},
            {0.0, 0.01},
            {0.0, 0.01},
            {-1, 1}
    };

    net.add_layer(l1);
    net.add_layer(l2);
    net.add_layer(l3);
    net.add_layer(l4);

    net.init(42, distribution_params);

    SGD opt;
    opt.m_decay = 0.8;
    opt.m_lrate = 0.01;

    net.train();
    net.fit(opt, train_data, train_label, 12, 3, 42);
    net.eval();

    std::cout << net << std::endl;

    Eigen::MatrixXd train_predict_before_save = net.predict(train_data);
    Eigen::MatrixXd test_predict_before_save = net.predict(test_data);

    std::cout << "Metrics in train data before saving..." << std::endl;
    metrics::regressor_metrics_calculate(train_label, train_predict_before_save);

    std::cout << "Metrics in test data before saving..." << std::endl;
    metrics::regressor_metrics_calculate(test_label, test_predict_before_save);

    net.export_net("test_saving_net", "model_1");

    NeuralNetwork net_;

    net_.read_net("test_saving_net", "model_1");

    std::cout << net_;

    net_.eval();
    Eigen::MatrixXd train_predict_after_save = net_.predict(train_data);
    Eigen::MatrixXd test_predict_after_save = net_.predict(test_data);

    std::cout << "Metrics in train data after saving..." << std::endl;
    metrics::regressor_metrics_calculate(train_label, train_predict_after_save);

    std::cout << "Metrics in test data after saving..." << std::endl;
    metrics::regressor_metrics_calculate(test_label, test_predict_after_save);
    return 0;
}