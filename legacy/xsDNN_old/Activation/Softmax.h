//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_SOFTMAX_H
#define XSDNN_SOFTMAX_H

namespace xsdnn {
    namespace activate {
        /*!
        \brief Класс функции активации - Softmax
        \author __[shuffle-true](https://github.com/shuffle-true)__
        \version 0.0
        \date Март 2022 года
        */
        class Softmax {
        private:
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
            static inline void activate(const xsTypes::Matrix &Z, xsTypes::Matrix &A) {
                // TODO: implement softmax forward
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
            static inline void apply_jacobian(const xsTypes::Matrix &Z, const xsTypes::Matrix &A,
                                              const xsTypes::Matrix &F, xsTypes::Matrix &G) {
                // TODO: implement softmax backward
            }

            ///
            /// \return Тип активации.
            static std::string return_type() {
                return "Softmax";
            }
        };
    } // namespace activate
} // namespace xsdnn

#endif //XSDNN_SOFTMAX_H
