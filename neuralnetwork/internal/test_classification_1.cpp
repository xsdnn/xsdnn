# include "../DNN.h"
# include <iostream>
typedef Eigen::MatrixXd Matrix;
typedef Eigen::RowVectorXd Vector;

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
		Matrix AxT = A * x.transpose();
		Matrix xAxT = x * AxT;
		Matrix xTb = x.transpose() * b;

		return Scalar(Scalar(0.5) * xAxT(0, 0) + xTb(0, 0));
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

void DATA_LOADER(Matrix& A, Vector& b, const int nobject)
{
	QuadraticOracle function(A, b);
	Matrix ONE = Eigen::MatrixXd::Ones(1, nobject);
	//TODO: реализовать подгрузку данных
}


int main()
{
	//TODO: написать архитектуру нейронки
	Matrix A(2, 2);
	Vector b(2);
	Vector x(2);

	A << 1, 0,
		 0, 1;
	b << 1, 1;
	x << -2, 0;
	QuadraticOracle obj(A, b);
	std::cout << obj.func(x);
	
	return 0;

	
}