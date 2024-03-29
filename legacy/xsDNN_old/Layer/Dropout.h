//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_INCLUDE_DROPOUT_H
#define XSDNN_INCLUDE_DROPOUT_H

namespace xsdnn {
    /*!
    \brief Класс Dropout слоя
    \author __[shuffle-true](https://github.com/shuffle-true)__
    \version 0.0
    \date Июль 2022 года
    */
    template<typename Activation>
    class Dropout : public Layer {
    public:
        /// Конструктор Dropout слоя
        /// \param in_size кол-во нейронов на вход - выход
        /// \param dropout_rate вероятность __отключения__ нейрона
        Dropout(const int &in_size, const Scalar &dropout_rate) :
                Layer(in_size, in_size, "undefined"),
                dropout_rate_(dropout_rate),
                scale_(Scalar(1.0) / (Scalar(1.0) - dropout_rate_)) {}

        void init(const std::vector<Scalar> &params, RNG &rng) override {}

        void init() override {}

        /// Проход вперед по слою
        ///
        /// 1. Значения нейронов предыдущего слоя поэлементно домножаются на маску распределения Бернулли
        /// 2. Получвшиеся значения нормируются с коэффицентом || 1 / (1 - p) ||
        /// \param prev_layer_data значения нейронов предыдущего слоя
        void forward(const xsTypes::Matrix &prev_layer_data) override {
            const long ncols = prev_layer_data.dimension(1);

            if (workflow == "train") {
                bernoulli.set_param(this->dropout_rate_, this->m_out_size);

                mask_.resize(this->m_out_size, ncols);
                m_z.resize(this->m_out_size, ncols);
                m_a.resize(this->m_out_size, ncols);

                for (int i = 0; i < ncols; i++) bernoulli(mask_.col(i).data());

                m_z.noalias() = prev_layer_data.cwiseProduct(mask_);
                m_z.noalias() = m_z * scale_;
                Activation::activate(m_z, m_a);
            } else {
                m_z.resize(this->m_out_size, ncols);
                m_z = prev_layer_data;
                Activation::activate(m_z, m_a);
            }
        }

        ///
        /// \return значения нейронов после прямого прохода
        const xsTypes::Matrix &output() const override { return m_a; }

        /// Обратный проход по слою
        ///
        /// \image html dropout_back_derivative.png
        ///
        /// Положение предыдущего - следующего слоя равносильно прямому проходу
        /// \param prev_layer_data выходы нейронов предыдущего слоя
        /// \param next_layer_data вектор градиента следующего слоя
        void backprop(const xsTypes::Matrix &prev_layer_data,
                      const xsTypes::Matrix &next_layer_backprop_data) override {
            const long ncols = prev_layer_data.cols();
            xsTypes::Matrix &dLz = m_z;
            Activation::apply_jacobian(m_z, m_a, next_layer_backprop_data, dLz);
            m_din.resize(this->m_out_size, ncols);
            m_din.noalias() = next_layer_backprop_data.cwiseProduct(mask_) * scale_;
            m_din = m_din.array() * dLz.array();
        }

        ///
        /// \return Вектор направления спуска (антиградиент)
        const xsTypes::Matrix &backprop_data() const override { return m_din; };

        /// В этом слое не участвуют алгоритмы оптимизации
        /// \param opt объект класса Optimizer
        void update(Optimizer &opt) override {}

        void train() override { workflow = "train"; }

        void eval() override { workflow = "eval"; }

        std::vector<Scalar> get_parametrs() const override { return std::vector<Scalar>(); }

        void set_parametrs(const std::vector<Scalar> &param) override {};

        std::vector<Scalar> get_derivatives() const override { return std::vector<Scalar>(); };

        std::string layer_type() const override { return "Dropout"; };

        std::string activation_type() const override { return Activation::return_type(); };

        std::string distribution_type() const override { return "undefined"; };

        void fill_meta_info(Meta &map, int index) const override {
            std::string ind = std::to_string(index);

            map.insert(std::make_pair("Layer " + ind, internal::layer_id(layer_type())));
            map.insert(std::make_pair("Activation " + ind, internal::activation_id(activation_type())));
            map.insert(std::make_pair("in_size " + ind, this->m_in_size));
            map.insert(std::make_pair("dropout_rate " + ind, this->dropout_rate_));
        };

        friend std::ostream &operator<<(std::ostream &out, Dropout &obj) {
            out << std::string(30, ':')
                << "Dropout Layer Information"
                << std::string(30, ':') << std::endl << std::endl;


            out << "In size: " << obj.m_in_size << " " << std::endl
                << "Dropout rate: " << obj.dropout_rate_ << " " << std::endl
                << "Workflow: " << obj.workflow << std::endl << std::endl;

            return out;
        }

        xsTypes::Matrix mask() {
            return mask_;
        }

    private:
        xsTypes::Matrix m_z;                ///< значения выходных нейронов до активации
        xsTypes::Matrix m_a;                ///< значения выходных нейронов после активации
        xsTypes::Matrix m_din;              ///< вектор градиента по этому слою
        xsTypes::Matrix mask_;              ///< маска, содержащая распределение Бернулли для отключения нейронов
        Scalar dropout_rate_;      ///< вероятность отключения нейронов (p)
        Scalar scale_;             ///< коэффицент масштабирования || 1 / (1 - p) ||
        internal::random::bernoulli bernoulli;          ///< заполнение маски слоя из распределения Бернулли

    };
} // namespace xsdnn


#endif //XSDNN_INCLUDE_DROPOUT_H
