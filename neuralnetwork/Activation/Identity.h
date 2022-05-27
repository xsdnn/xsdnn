#pragma once

# include <Eigen/Core>
# include "../Config.h"

class Identity
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:

	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.array() = Z.array();
	}

	static inline void apply_jacobian(const Matrix& Z, const Matrix& A,
		const Matrix& F, Matrix& G)
	{
		G.array() = F.array();
	}

	static std::string return_type()
	{
		return "Identity";
	}
};
