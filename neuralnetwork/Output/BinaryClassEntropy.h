#pragma once


# include <Eigen/Core>
# include <stdexcept>
# include "../Config.h"
# include "../Output.h"

///
///	Блок управления задачей бинарной классификации на критерии кросс-энтропии
/// 


class BinaryClassEntropy : public Output
{
private:
	Matrix m_din;			// производные по выходному слою

public:
	void check_target_data(const Matrix& target) const
	{
		const int nelem = target.size();
		const Scalar* target_data = target.data();

		for (int i = 0; i < nelem; i++)
		{
			if (target_data[i] != Scalar(1) && target_data[i] != Scalar(0)) 
			{
				throw std::invalid_argument("[class BinaryClassEntropy]: target data is not 1 and 0. Check input param!");
			}
		}
	}

	void check_target_data(IntegerVector& target) const
	{
		const int nelem = target.size();

		for (int i = 0; i < nelem; i++)
		{
			if (target[i] != Scalar(1) && target[i] != Scalar(0))
			{
				throw std::invalid_argument("[class BinaryClassEntropy]: target data is not 1 and 0. Check input param!");
			}
		}
	}

	void evaluate(const Matrix& prev_layer_data, const Matrix& target)
	{
		const int ncols = prev_layer_data.cols();
		const int nrows = prev_layer_data.rows();

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


	void evaluate(const Matrix& prev_layer_data, const IntegerVector& target)
	{
		const int nrows = target.rows();

		if (nrows != 1)
		{
			throw std::invalid_argument("[class BinaryClassEntropy] Target rows != 1. Check input.");
		}

		const int ncols = prev_layer_data.cols();

		if (target.size() != ncols)
		{
			throw std::invalid_argument("[class BinaryClassEntropy]: Target data have incorrect dim. Check input data");
		}

		m_din.resize(1, ncols);
		m_din.array() = (target.array() == Scalar(0)).select((Scalar(1) - prev_layer_data.array()).cwiseInverse(),
			-prev_layer_data.array().cwiseInverse());
	}

	const Matrix& backprop_data() const
	{
		return m_din;
	}

	Scalar loss() const
	{
		//	Зная m_din, подставим его в лосс и выразим ошибку.
		//	Получим ->
		//	L = E( log(|m_din|) ) / N

		return Scalar(m_din.array().abs().log().sum()) / m_din.cols();
	}

	std::string output_type() const
	{
		return "BinaryClassEntropy";
	}

};
