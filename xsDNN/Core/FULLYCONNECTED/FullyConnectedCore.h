//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_FULLYCONNECTEDCORE_H
#define XSDNN_FULLYCONNECTEDCORE_H

#include "FullyConnected_DIRECT.h"
namespace xsdnn {
    namespace internal {
        namespace fc {
            /// Прямое распространение полносвязного слоя
            /// \tparam Activation функция активации
            /// \param prev_data выходы предыдущего слоя
            /// \param w матрица весов
            /// \param b вектор смещений
            /// \param z значения нейронов текущего слоя до активации
            /// \param a значения нейронов текущего слоя после активации
            /// \param bias использовать смещение?
            /// \param out_size output dimension
            template<typename Activation>
            inline void computeForward(const xsTypes::Matrix& prev_data,
                                       xsTypes::Matrix& w,
                                       xsTypes::Vector& b,
                                       xsTypes::Matrix& z,
                                       xsTypes::Matrix& a,
                                       bool    bias,
                                       const int out_size) {
                internal::fc::algorithm::compute_forward_direct<Activation>(prev_data,
                                                                            w,
                                                                            b,
                                                                            z,
                                                                            a,
                                                                            bias,
                                                                            out_size);
            }

            /// Обратное распространение полносвязного слоя
            /// \tparam Activation функция активации
            /// \param prev_data значения нейронов предыдущего слоя
            /// \param next_grad значения градиентов следующего слоя
            /// \param w матрица весов
            /// \param dw матрица производных весов
            /// \param db вектор производных смещений
            /// \param din матрица производных выходов нейронов
            /// \param z значения нейронов до активации
            /// \param a значения нейронов после активации
            /// \param bias применять смещение?
            /// \param in_size входной размер текущего слоя
            template<typename Activation>
            inline void computeBackward(const xsTypes::Matrix& prev_data,
                                        const xsTypes::Matrix& next_grad,
                                        xsTypes::Matrix& w,
                                        xsTypes::Matrix& dw,
                                        xsTypes::Vector& db,
                                        xsTypes::Matrix& din,
                                        xsTypes::Matrix& z,
                                        xsTypes::Matrix& a,
                                        bool    bias,
                                        const int in_size) {
                internal::fc::algorithm::compute_backward_direct<Activation>(
                        prev_data,
                        next_grad,
                        w,
                        dw,
                        db,
                        din,
                        z,
                        a,
                        bias,
                        in_size
                );
            }
        } // namespace fc
    } // namespace internal
} // namespace xsdnn


#endif //XSDNN_FULLYCONNECTEDCORE_H
