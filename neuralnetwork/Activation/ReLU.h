#pragma once

# include <Eigen/Core>
# include "../Config.h"

class ReLU
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:

	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.array() = Z.array().cwiseMax(Scalar(0));
	}

	static inline void apply_jacobian(const Matrix& Z, const Matrix& A,
		const Matrix& F, Matrix& G)
	{
		G.array() = (A.array() > Scalar(0)).select(F, Scalar(0));
	}

	static std::string return_type()
	{
		return "ReLU";
	}
};