#pragma once

# include <Eigen/Core>
# include <stdexcept>
# include "../Config.h"
# include "../Output.h"

///
/// Блок управления выходным слоем при решении задачи регрессии на ошибке MSE
/// 




class RegressionMSE : public Output
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_din; // Производная от входных данных этого слоя

public:
	void evaluate(const Matrix& prev_layer_data, const Matrix& target)
	{
		const int ncol = prev_layer_data.cols();
		const int nrow = prev_layer_data.rows();

		// Проверяем соотвествие правилу вход == выход предыдущего
		if ((target.cols() != ncol) || (target.rows() != nrow))
		{
			throw std::invalid_argument("[class RegressionMSE]: Target data have incorrect dim. Check input data");
		}

		// собственно делаем расчет

		m_din.resize(nrow, ncol);
		m_din.noalias() = prev_layer_data - target;
	}

	const Matrix& backprop_data() const
	{
		return m_din;
	}

	Scalar loss() const
	{
		return Scalar(0.5) * m_din.squaredNorm() / m_din.cols();
	}

	std::string output_type() const
	{
		return "RegressionMSE";
	}
};