#pragma once

# include <Eigen/Core>
# include "../Config.h"

/*!
 * \details Класс функции активации - Softmax.
 */
class Softmax
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> RowArray;

public:

    /// __Алгоритм__:
    /// \code
    /// int softmax_forward(const Matrix& Z, Matrix& A){
    ///
    ///     for (int i = 0; i < Z.size(); i++)
    ///     {
    ///         A[i] = Z[i] * ( 1 / exp(Z).sum() );
    ///     }
    ///
    /// }
    /// \endcode
    /// \param Z значения нейронов до активации
    /// \param A значения нейронов после активации
	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.array() = (Z.rowwise() - Z.colwise().maxCoeff()).array().exp();
		RowArray colsums = A.colwise().sum();
		A.array().rowwise() /= colsums;
	}

    /// Операция матричного дифференцирования.

    /// __Алгоритм__:
    /// \code
    /// int softmax_backprop(const Matrix& Z, const Matrix& A,
    ///		const Matrix& F, Matrix& G) {
    ///
    ///     for (int i = 0; i < A.size(); i++)
    ///     {
    ///         G = A * (F - transpose(A));
    ///     }
    ///
    /// }
    /// \endcode
    /// \param Z нейроны слоя до активации.
    /// \param A нейроны слоя после активации.
    /// \param F нейроны следующего слоя.
    /// \param G значения, которые получаются после backprop.
	static inline void apply_jacobian(const Matrix& Z, const Matrix& A,
		const Matrix& F, Matrix& G)
	{
		RowArray a_dot_f = A.cwiseProduct(F).colwise().sum();
		G.array() = A.array() * (F.array().rowwise() - a_dot_f);
	}

    ///
    /// \return Тип активации.
	static std::string return_type()
	{
		return "Softmax";
	}
};