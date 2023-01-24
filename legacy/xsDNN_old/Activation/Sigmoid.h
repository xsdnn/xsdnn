//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_SIGMOID_H
#define XSDNN_SIGMOID_H

namespace xsdnn {
    namespace activate {
        /*!
        \brief Класс функции активации - Sigmoid
        \author __[shuffle-true](https://github.com/shuffle-true)__
        \version 0.0
        \date Март 2022 года
        */
        class Sigmoid {
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
            static inline void activate(const xsTypes::Matrix &Z, xsTypes::Matrix &A) {
                A = Scalar(1) / (Scalar(1) + (-Z).exp()).eval();
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
            static inline void apply_jacobian(const xsTypes::Matrix &Z, const xsTypes::Matrix &A,
                                              const xsTypes::Matrix &F, xsTypes::Matrix &G) {
                G = A * (Scalar(1) - A) * F;
            }

            ///
            /// \return Тип активации.
            static std::string return_type() {
                return "Sigmoid";
            }
        };
    } // namespace activate
} // namespace xsdnn


#endif //XSDNN_SIGMOID_H
