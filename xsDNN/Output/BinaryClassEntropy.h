//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

/*!
\brief Класс функционала ошибки для бинарной классификации (BinaryClassEntropy)
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Май 2022 года
*/
class BinaryEntropyLoss : public Output
{
private:
	Matrix m_din;			///< производные по выходному слою

public:
    /// Проверка входных данных на соотвествие значений ---  1 || 0
    /// \param target целевая переменная
	void check_target_data(const Matrix& target) const override
	{
		const long nelem = target.size();
		const Scalar* target_data = target.data();

		for (int i = 0; i < nelem; i++)
		{
			if (target_data[i] != Scalar(1) && target_data[i] != Scalar(0)) 
			{
				throw std::invalid_argument("[class BinaryClassEntropy]: target data is not 1 and 0. Check input param!");
			}
		}
	}

    /// \image html binaryclassentropy_evaluate.png
    /// \param prev_layer_data последний скрытый слой сети
    /// \param target целевая переменная
	void evaluate(const Matrix& prev_layer_data, const Matrix& target) override
	{
		const long ncols = prev_layer_data.cols();
		const long nrows = prev_layer_data.rows();

		if ((target.cols() != ncols) || (target.rows() != nrows))
		{
			throw std::invalid_argument("[class BinaryClassEntropy]: Target data have incorrect dim. Check input data");
		}

		//	L = -y * log(in) - (1 - y) * log(1 - in)
		//  in = phat
		//  d (L) / d(in) = -(y) / phat + (1 - y) / (1 - phat)
		
		m_din.resize(nrows, ncols);
		
		//	Если 'y' = 0 ->  d (L) / d(in) = (1) / (1 - phat)
		//	Если 'y' = 1 ->  d (L) / d(in) = - (1) / phat

		m_din.array() = (target.array() == Scalar(0)).select((Scalar(1) - prev_layer_data.array()).cwiseInverse(),
			-prev_layer_data.array().cwiseInverse());
	}

    ///
    /// \return Вектор направления спуска (антиградиент).
	const Matrix& backprop_data() const override
	{
		return m_din;
	}

    /// \image html binaryclassentropy_loss.png
    /// \return Ошибку на обучающей выборке
	Scalar loss() const override
	{
		//	Зная m_din, подставим его в лосс и выразим ошибку.
		//	Получим ->
		//	L = E( log(|m_din|) ) / N

		return Scalar(m_din.array().abs().log().sum()) / static_cast<Scalar>(m_din.cols());
	}

    ///
    /// \return Тип выходного слоя.
	std::string output_type() const override
	{
		return "BinaryClassEntropy";
	}
};
