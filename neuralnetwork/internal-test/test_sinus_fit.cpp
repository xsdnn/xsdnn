# include "../DNN.h"
# include <iostream>
# include <vector>
# include <ctime>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

int main()
{
  // generate train and test data
  Matrix train_data = Eigen::MatrixXd::Random(1, 20000) * 3.1415f;
  Matrix train_target = train_data.array().sin();
  Matrix test_data = Eigen::MatrixXd::Random(1, 1000) * 6.2832f;
  Matrix test_target = test_data.array().sin();

  // define NeuralNetwork class
  NeuralNetwork net;

  // define layer's
	Layer* l1 = new FullyConnected<ReLU>(1, 10);
	Layer* l2 = new FullyConnected<ReLU>(10, 10);
	Layer* l3 = new FullyConnected<Identity>(10, 1);

  // add layer's
	net.add_layer(l1);
	net.add_layer(l2);
	net.add_layer(l3);

  // set regressor callback and mse output
	VerboseCallback callback;
	net.set_callback(callback);
	net.set_output(new RegressionMSE());

  // define stochastic gradient descent optimizer
	SGD opt;
  opt.m_lrate = 0.01;

  // init weight and biases normal distribution with mean = 0 and varience = 0.1
	net.init(0, 0.1, 123);

  std::cout << "Net init - start training" << std::endl;
  // start learning net with batch_size = 250, nums epoch = 5000 and random seed = 123
  unsigned int start_time = clock();
	net.fit(opt, train_data, train_target, 10, 25, 123);
  unsigned int end_time = clock();

  std::cout << "Training completed, elapsed time = " << (end_time - start_time) / 1000000.0 << " сек." << std::endl;

  // find max error (MAE) on test choice
  Matrix predict = net.predict(test_data);

  MSE_calculate(test_target, predict);
  MAE_calculate(test_target, predict);
  R_calculate(test_target, predict);
  MAPE_calculate(test_target, predict);
  return 0;
}
