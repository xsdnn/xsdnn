//
// Created by shuffle on 03.08.22.
//

#ifndef XSDNN_INCLUDE_LEAKYRELU_H
#define XSDNN_INCLUDE_LEAKYRELU_H

namespace activate{
    /*!
    \brief Класс функции активации - LeakyReLU
    \author __[shuffle-true](https://github.com/shuffle-true)__
    \version 0.0
    \date Август 2022 года
    */
    class LeakyReLU{
    public:
        constexpr static Scalar _negative_slope = 0.01;             ///< см. алгоритм

        /// __Алгоритм__:
        /// \code
        /// int leaky_relu_forward(const Matrix& Z, Matrix& A){
        ///
        ///     for (int i = 0; i < Z.size(); i++)
        ///     {
        ///         A[i] = Z[i] > 0 ? Z[i] : negative_slope * Z[i];
        ///     }
        ///
        /// }
        /// \endcode
        /// \param Z значения нейронов до активации
        /// \param A значения нейронов после активации
        static inline void activate(const Matrix& Z, Matrix& A)
        {
            A.array() = (Z.array() > Scalar(0)).select(Z, _negative_slope * Z);
        }


        /// Операция матричного дифференцирования.

        /// __Алгоритм__:
        /// \code
        /// int relu_backprop(const Matrix& Z, const Matrix& A,
        ///		const Matrix& F, Matrix& G) {
        ///
        ///     for (int i = 0; i < A.size(); i++)
        ///     {
        ///         G[i] = A[i] > 0 ? F[i] : negative_slope * A[i];
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
            G.array() = (A.array() > Scalar(0)).select(F, _negative_slope * A.array());
        }

        ///
        /// \return Тип активации.
        static std::string return_type()
        {
            return "LeakyReLU";
        }
    };
}

#endif //XSDNN_INCLUDE_LEAKYRELU_H
