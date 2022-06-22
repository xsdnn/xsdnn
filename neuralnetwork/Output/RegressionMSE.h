#pragma once

# include <Eigen/Core>
# include <stdexcept>
# include "../Config.h"
# include "../Output.h"


/*!
 * Класс управления задачей регрессии - критерий MSE
 */
class RegressionMSE : public Output
{
private:
	Matrix m_din; // Производная от входных данных этого слоя

public:
    /// Проверка входных данных на соотвествие размерам
    /// \param prev_layer_data последний скрытый слой сети
    /// \param target целевая переменная
	void evaluate(const Matrix& prev_layer_data, const Matrix& target) override
	{
		const long ncol = prev_layer_data.cols();
		const long nrow = prev_layer_data.rows();

		// Проверяем соотвествие правилу вход == выход предыдущего
		if ((target.cols() != ncol) || (target.rows() != nrow))
		{
			throw std::invalid_argument("[class RegressionMSE]: Target data have incorrect dim. Check input data");
		}

		// собственно делаем расчет

		m_din.resize(nrow, ncol);
		m_din.noalias() = prev_layer_data - target;
	}

    ///
    /// \return Вектор направления спуска (антиградиент).
	const Matrix& backprop_data() const override
	{
		return m_din;
	}

    ///
    /// \return Ошибку на обучающей выборке
	Scalar loss() const override
	{
		return Scalar(0.5) * m_din.squaredNorm() / static_cast<Scalar>(m_din.cols());
	}

    ///
    /// \return Тип выходного слоя.
	std::string output_type() const override
	{
		return "RegressionMSE";
	}
};