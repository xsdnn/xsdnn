//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_LAYER_H
#define XSDNN_LAYER_H

# include "Optimizer.h"

namespace xsdnn {
    /*!
    \brief Родительский класс слоя
    \author __[shuffle-true](https://github.com/shuffle-true)__
    \version 0.0
    \date Март 2022 года
    */
    class Layer {
    protected:
        typedef std::map<std::string, Scalar> Meta;

        const int m_in_size;
        const int m_out_size;
        std::string workflow;


    public:
        Layer(const int in_size, const int out_size, std::string init_workflow) :
                m_in_size(in_size), m_out_size(out_size), workflow(init_workflow) {}

        virtual ~Layer() = default;


        /// <summary>
        /// None
        /// </summary>
        /// <returns>Кол-во входящих нейронов</returns>
        int in_size() const {
            return m_in_size;
        }

        /// <summary>
        /// None
        /// </summary>
        /// <returns>Кол-во выходящих нейронов</returns>
        int out_size() const {
            return m_out_size;
        }

        /// Получить текущий процесс
        std::string get_workflow() const {
            return workflow;
        }

        /// Инициализация слоя
        /// \param params вектор параметров для конкретного распределения
        /// \param batch_size размер батча
        /// \param rng ГСЧ
        virtual void init(const std::vector<Scalar> &params, RNG &rng) = 0;

        /// <summary>
        /// Инициализация слоя без входных параметров,
        /// будет использоваться при чтении сетки из файла
        /// </summary>
        virtual void init() = 0;

        /// <summary>
        /// Метод прохода вперед по сетке.
        ///
        /// Будет инициализировано n слоев, у каждого слоя есть такой метод,
        /// они будут поочередно вызываться из каждого слоя и таким образом,
        /// мы получим полный проход по всей сетке.
        ///
        /// Реализация метода отличается у каждого типа слоев
        /// </summary>
        /// <param name="prev_layer_data"> - предыдущий слой, значения его нейронов</param>
        virtual void forward(const xsTypes::Matrix &prev_layer_data) = 0;

        /// <summary>
        /// None
        /// </summary>
        /// <returns>Значения нейронов в слою после функции активации. Использование
        /// метода разрешено только после метода Layer::forward()!</returns>
        virtual const xsTypes::Matrix &output() const = 0;

        /// Обратный проход по сети
        /// \param prev_layer_data - output слоя слева
        /// \param next_layer_backprop_data - backprop_data слоя справа
        virtual void backprop(const xsTypes::Matrix &prev_layer_data,
                              const xsTypes::Matrix &next_layer_backprop_data) = 0;

        /// <summary>
        /// None
        /// </summary>
        /// <returns>Возвращает next_layer_data в Layer::backprop()
        /// </returns>
        virtual const xsTypes::Matrix &backprop_data() const = 0;

        /// <summary>
        /// Обновление параметров сетки после обратного распространения
        /// </summary>
        /// <param name="opt"> - алгоритм оптимизации градиента</param>
        virtual void update(Optimizer &opt) = 0;

        /// Установить рабочий процесс - тренировка
        virtual void train() = 0;

        /// Установить рабочий процесс - тестирование
        virtual void eval() = 0;

        virtual std::vector<Scalar> get_parametrs() const = 0;

        virtual void set_parametrs(const std::vector<Scalar> &param) {};

        virtual std::vector<Scalar> get_derivatives() const = 0;

        virtual std::string layer_type() const = 0;

        virtual std::string activation_type() const = 0;

        virtual std::string distribution_type() const = 0;

        /// <summary>
        /// Метод нужен для заполнения основной информации о слое -
        /// тип слоя,
        /// входные и выходные значения кол-ва нейронов и т.д.
        /// Нужен для импорта сетки в файл
        /// </summary>
        /// <param name="map"> - словарь</param>
        /// <param name="index"> - индекс слоя в модели</param>
        virtual void fill_meta_info(Meta &map, int index) const = 0;
    };
} // namespace xsdnn


#endif //XSDNN_LAYER_H
