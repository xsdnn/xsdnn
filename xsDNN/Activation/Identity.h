//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

namespace activate{
    /*!
	\brief Класс функции активации - Identity
    \author __[shuffle-true](https://github.com/shuffle-true)__
	\version 0.0
	\date Март 2022 года
    */
    class Identity
    {
    public:
        /// __Алгоритм__:
        /// \code
        /// int identity_forward(const Matrix& Z, Matrix& A){
        ///
        ///     A = Z;
        ///
        /// }
        /// \endcode
        /// \param Z значения нейронов до активации
        /// \param A значения нейронов после активации
        static inline void activate(const Matrix& Z, Matrix& A)
        {
            A.array() = Z.array();
        }

        /// Операция матричного дифференцирования.

        /// __Алгоритм__:
        /// \code
        /// int identity_backprop(const Matrix& Z, const Matrix& A,
        ///		const Matrix& F, Matrix& G) {
        ///
        ///     G = F;
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
            G.array() = F.array();
        }

        ///
        /// \return Тип активации.
        static std::string return_type()
        {
            return "Identity";
        }
    };
}

