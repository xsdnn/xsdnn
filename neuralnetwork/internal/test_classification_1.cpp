# include "../DNN.h"
# include <iostream>
# include <vector>
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

//#####################################################################################################
//#								Тестируется квадратичная функция.								      #
//#			Как обучается нейросеть на определение лежит ли точка внутри функции или снаружи.         #
//#####################################################################################################

class QuadraticOracle
{
private:
	Matrix A;
	Vector b;

public:
	QuadraticOracle(Matrix& _A, Vector& _b) :
		A(_A), b(_b) {}

	void input_data(Matrix& _A, Vector& _b)
	{
		A = _A;
		b = _b;
	}

	Scalar func(Vector& x)
	{
		Matrix Ax = A * x;
		Matrix xTAx = x.transpose() * Ax;
		Matrix xTb = x.transpose() * b;

		return Scalar(Scalar(0.5) * xTAx(0, 0) + xTb(0, 0));
	}

	friend std::ostream& operator << (std::ostream& output, QuadraticOracle& obj)
	{
		output << "A is ->" << std::endl;
		output << obj.A << std::endl << std::endl;
		output << "b is ->" << std::endl;
		output << obj.b << std::endl << std::endl;
		return output;
	}
};

void DATA_LOADER(Matrix& A, Vector& b, Matrix& train_data, Matrix& train_target, Matrix& test_data, Matrix& test_target, const int nobject)
{
	QuadraticOracle function(A, b);
	Matrix ONE = Eigen::MatrixXd::Ones(1, nobject);

	train_data = Eigen::MatrixXd::Random(2, nobject) * Scalar(1);
	test_data = Eigen::MatrixXd::Random(2, nobject) * Scalar(1);

	Matrix train_target_before = Eigen::MatrixXd::Random(1, nobject) * Scalar(18);
	Matrix test_target_before = Eigen::MatrixXd::Random(1, nobject) * Scalar(9);

	// вычисляем значение функции в каждой точке тренировочного набора
	train_target.resize(1, nobject);
	for (int i = 0; i < nobject; i++)
	{
		Vector col = train_data.col(i);
		Scalar quadratic_oracle_func = function.func(col);
		train_target.array() = (train_target_before.array() >= quadratic_oracle_func).select(ONE, Scalar(0));
	}

	// вычисляем значение функции в каждой точке тестового набора
	test_target.resize(1, nobject);
	for (int i = 0; i < nobject; i++)
	{
		Vector col = test_data.col(i);
		Scalar quadratic_oracle_func = function.func(col);
		test_target.array() = (test_target_before.array() >= quadratic_oracle_func).select(ONE, Scalar(0));
	}
}


int main()
{
	Matrix A(2, 2);
	Vector b(2);
	const int nobject = 1000;
	A << 1, 0,
		 0, 1;
	b << 1, 1;
	
	Matrix train_data;
	Matrix train_target;
	Matrix test_data;
	Matrix test_target;

	DATA_LOADER(A, b, train_data, train_target, test_data, test_target, nobject);
	
	//std::cout << train_data << std::endl;
	//std::cout << train_target << std::endl;
	//std::cout << test_data << std::endl;
	//std::cout << test_target << std::endl;

	// Архитектура нейросети

	NeuralNetwork net;

	Layer* l1 = new FullyConnected<Identity>(2, 2);
	Layer* l2 = new FullyConnected<Identity>(2, 3);
	Layer* l3 = new FullyConnected<Identity>(3, 2);
	Layer* l4 = new FullyConnected<Sigmoid>(2, 1);

	net.add_layer(l1);
	net.add_layer(l2);
	net.add_layer(l3);
	net.add_layer(l4);

	CustomClassificationCallback callback;
	net.set_callback(callback);
	net.set_output(new BinaryClassEntropy());
	
	SGD opt;

	net.init(0, 0.1, 123);
	net.fit(opt, train_data, train_target, 1000, 1000, 123);
	return 0;
}