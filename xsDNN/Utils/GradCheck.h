//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_GRAD_CHECK_H
#define XSDNN_GRAD_CHECK_H

#include <limits>
#include <cmath>

namespace internal {
    namespace debug {
        /// Численный подсчет градиента по формуле
        /// https://cs231n.github.io/neural-networks-3/#gradcheck
        /// \param layer слой, на котором происходит тестирование
        /// \param in_data входной вектор данных, состоящий из одного объекта
        /// \param in_pos позиция на входе для подсчета производной
        /// \param out_pos позиция на выходе для подсчета производной
        /// \return производная по заданной точке
        inline Scalar numerical_gradient(
                Layer* layer,
                Matrix& in_data,
                const int in_pos,
                const int out_pos
        )
        {
            Scalar h = std::sqrt(std::sqrt(std::numeric_limits<Scalar>::epsilon()));
            Scalar prev_in = in_data(in_pos, 0);
            in_data(in_pos, 0) = prev_in + h;
            layer->forward(in_data);
            Scalar out1 = layer->output()(out_pos, 0);
            in_data(in_pos, 0) = prev_in - h;
            layer->forward(in_data);
            Scalar out2 = layer->output()(out_pos, 0);
            return (out1 - out2) / (2 * h);
        }

        /// Подсчет градиента через обратное распространение
        /// \param layer слой, на котором происходит тестирование
        /// \param in_data входной вектор данных, состоящий из одного объекта
        /// \param in_pos позиция на входе для подсчета производной
        /// \param out_pos позиция на выходе для подсчета производной
        /// \return производная по заданной точке
        inline Scalar analytical_gradient(
                Layer* layer,
                Matrix& in_data,
                const int in_pos,
                const int out_pos,
                const int out_size
        )
        {
            const int in_size = in_data.rows();
            Matrix next_layer_backprop_data(out_size, 1); next_layer_backprop_data.setZero();
            next_layer_backprop_data(out_pos, 0) = Scalar(1.0);
            Matrix prev_layer_data(in_size, 1); prev_layer_data.setZero();
            layer->forward(in_data);
            layer->backprop(prev_layer_data, next_layer_backprop_data);
            return layer->backprop_data()(in_pos, 0);
        }
    } // end namespace debug
} // end namespace internal

#endif //XSDNN_GRAD_CHECK_H
