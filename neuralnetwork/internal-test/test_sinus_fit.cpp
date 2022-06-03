# include "../DNN.h"
# include <iostream>
# include <vector>
# include <cmath>
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

int main()
{
  // generate train and test data
  Matrix train_data = Eigen::MatrixXd::Random(1, 1000) * 3.1415f;
  Matrix train_target = train_data.array().sin();
  Matrix test_data = Eigen::MatrixXd::Random(1, 5) * 3.1415f;
  Matrix test_target = test_data.array().sin();

  // define NeuralNetwork class
  NeuralNetwork net;

  // define layer's
	Layer* l1 = new FullyConnected<Sigmoid>(1, 10);
	Layer* l2 = new FullyConnected<Sigmoid>(10, 10);
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
	net.fit(opt, train_data, train_target, 250, 5000, 123);
  std::cout << "Training completed" << std::endl;

  // find max error (MAE) on test choice
  Matrix error = (net.predict(test_data) - test_target).array().cwiseAbs();
  std::cout << "Max Error MAE = " << error.maxCoeff() << std::endl;
  std::cout << "Min Error MAE = " << error.minCoeff() << std::endl;
  std::cout << "NeuralNetwork output = " << std::endl << net.predict(test_data) << std::endl;
  std::cout << "Real output = " << std::endl << test_target << std::endl;
  return 0;
}
