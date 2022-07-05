#pragma once

# include <Eigen/Core>
# include "../Config.h"

namespace activate{
    /*!
    * \details Класс функции активации - Sigmoid.
    */
    class Sigmoid
    {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    public:

        /// __Алгоритм__:
        /// \code
        /// int sigmoid_forward(const Matrix& Z, Matrix& A){
        ///
        ///     for (int i = 0; i < Z.size(); i++)
        ///     {
        ///         A[i] = 1 / (1 + exp(-Z[i]));
        ///     }
        ///
        /// }
        /// \endcode
        /// \param Z значения нейронов до активации
        /// \param A значения нейронов после активации
        static inline void activate(const Matrix& Z, Matrix& A)
        {
            A.array() = Scalar(1) / (Scalar(1) + (-Z.array()).exp());
        }

        /// Операция матричного дифференцирования.

        /// __Алгоритм__:
        /// \code
        /// int sigmoid_backprop(const Matrix& Z, const Matrix& A,
        ///		const Matrix& F, Matrix& G) {
        ///
        ///     for (int i = 0; i < A.size(); i++)
        ///     {
        ///         G[i] = A[i] * (1 - A[i]) * F[i];
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
            G.array() = A.array() * (Scalar(1) - A.array()) * F.array();
        }

        ///
        /// \return Тип активации.
        static std::string return_type()
        {
            return "Sigmoid";
        }
    };
}
