//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

/*!
\brief Класс функционала ошибки для регрессии (RegressionMSE)
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Март 2022 года
*/
class MSELoss : public Output
{
private:
	Matrix m_din; ///< Производная от входных данных этого слоя

public:
    /// \image html regressionmse_evaluate.png
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

    /// \image html regressionmse_loss.png
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