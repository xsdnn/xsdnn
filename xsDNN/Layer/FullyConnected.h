//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_FULLYCONNECTED_H
#define XSDNN_FULLYCONNECTED_H

#include "../Utils/Except.h"
#include "../Core/FULLYCONNECTED/FullyConnectedCore.h"

namespace xsdnn {
    /*!
    \brief Класс FullyConnected слоя
    \author __[shuffle-true](https://github.com/shuffle-true)__
    \version 0.0
    \date Июнь 2022 года
    */
    template<typename Distribution, typename Activation>
    class FullyConnected : public Layer {
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Vector::AlignedMapType AlignedMapVec;

        Matrix m_weight;         ///< Веса модели
        Vector m_bias;           ///< Смещение весов
        Matrix m_dw;             ///< Производная весов
        Vector m_db;             ///< Производная смещения
        Matrix m_z;              ///< Значения нейронов до активации
        Matrix m_a;              ///< Значения нейронов после активации
        Matrix m_din;            ///< Проивзодная значений нейронов после backprop

        bool BIAS_ACTIVATE;    ///< Применять смещение?


    public:
        /// Конструктор полносвязного слоя
        /// \param in_size кол-во нейронов на вход.
        /// \param out_size кол-во нейронов на выход.
        /// \param bias_true_false включить / выключить смещения весов. По умолч. включены = true.
        explicit FullyConnected(const int in_size, const int out_size, bool bias = true) :
                Layer(in_size, out_size, "undefined"), BIAS_ACTIVATE(bias) {}

        void init(const std::vector<Scalar> &params, RNG &rng) override {
            init();

            Distribution::set_random_data(m_weight.data(), m_weight.size(), rng, params);
            if (BIAS_ACTIVATE) { Distribution::set_random_data(m_bias.data(), m_bias.size(), rng, params); }

            //std::cout << "Weights " << std::endl << m_weight << std::endl;
            //std::cout << "Bias " << std::endl << m_bias << std::endl;
        }

        void init() override {
            m_weight.resize(this->m_in_size, this->m_out_size);
            m_dw.resize(this->m_in_size, this->m_out_size);
            if (BIAS_ACTIVATE) {
                m_bias.resize(this->m_out_size);
                m_db.resize(this->m_out_size);
            }
        }

        /// Описание того, как толкаются данные по сетке внутри одного слоя.
        /// Сначала получаем значения нейронов просто перемножая веса и предыдущие значения нейронов.
        /// Добавляем к этому смещение, если необходимо.
        /// Активируем.
        /// \param prev_layer_data матрица значений нейронов предыдущего слоя
        void forward(const Matrix &prev_layer_data) override {

            internal::fc::computeForward<Activation>(
                    prev_layer_data,
                    m_weight,m_bias,
                    m_z,m_a,
                    BIAS_ACTIVATE,this->m_out_size
            );

        }

        ///
        /// \return Значения нейронов после активации
        const Matrix &output() const override { return m_a; }

        /// Получаем производные этого слоя.
        /// Нужно получить производные по 3 вещам.
        /// 1. Производные весов - считаем Якобиан и умножаем на предыдущий слой
        /// 2. Производные смещения - среднее по строкам производных весов
        /// 3. Производные текущих значений нейронов - текущий вес на Якобиан.
        /// Положение предыдущего - следующего слоя равносильно прямому проходу
        /// \param prev_layer_data значения нейронов предыдущего слоя
        /// \param next_layer_data вектор градиента следующего слоя (слева - направо)
        void backprop(const Matrix &prev_layer_data,
                      const Matrix &next_layer_backprop_data) override {

            internal::fc::computeBackward<Activation>(
                    prev_layer_data, next_layer_backprop_data,
                    m_weight, m_dw,
                    m_db, m_din,
                    m_z, m_a,
                    BIAS_ACTIVATE, this->m_in_size
            );

        }

        ///
        /// \return Вектор направления спуска (антиградиент)
        const Matrix &backprop_data() const override { return m_din; }


        /// Обновление весов и смещений используя переданный алгоритм оптимизации (см. Optimizer)
        /// \param opt оптимайзер. Объект класса, унаследованного от Optimizer.
        void update(Optimizer &opt) override {
            AlignedMapVec dw(m_dw.data(), m_dw.size());
            AlignedMapVec w(m_weight.data(), m_weight.size());
            AlignedMapVec db(m_db.data(), m_db.size());
            AlignedMapVec b(m_bias.data(), m_bias.size());

            opt.update(dw, w);
            if (BIAS_ACTIVATE) { opt.update(db, b); }

        }

        ///
        /// \return Параметры слоя. Со смещением или без взависимости от условия.
        std::vector<Scalar> get_parametrs() const override {
            if (BIAS_ACTIVATE) {
                std::vector<Scalar> res(m_weight.size() + m_bias.size()); // указали кол-во ячеек в этом векторе
                // просто копируем в этот массив все содержимое
                std::copy(m_weight.data(), m_weight.data() + m_weight.size(),
                          res.begin()); // все аргументы передаются в виде указателей. Откуда начинаем, где заканчиваем, куда начинаем ложить.
                std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_weight.size());
                return res;
            } else {
                std::vector<Scalar> res(m_weight.size());
                std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin());
                return res;
            }

        }

        void train() override {
            workflow = "train";
        }

        void eval() override {
            workflow = "eval";
        }

        /// Установить параметры сетки. Используется при загрузки сети из файла.
        /// \param param 1-D вектор параметров.
        void set_parametrs(const std::vector<Scalar> &param) override {
            if (BIAS_ACTIVATE) {
                // сделаем проверку на равенство длин массивов
                // static_cast<int> - приведение длины массива к интовому типу данных
                if (static_cast<int>(param.size()) != (m_weight.size() + m_bias.size())) {
                    throw internal::except::xs_error(
                            "[class FullyConnected]: Parameter size does not match. Check parameter size!");
                }

                // если размеры сходятся, то копируем переданные значения в наши массивы
                std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data());
                std::copy(param.begin() + m_weight.size(), param.end(), m_bias.data());
            } else {
                if (static_cast<int>(param.size()) != (m_weight.size())) {
                    throw internal::except::xs_error(
                            "[class FullyConnected]: Parameter size does not match. Check parameter size!");
                }

                // если размеры сходятся, то копируем переданные значения в наши массивы
                std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data());
            }
        }

        ///
        /// \return Производные весов и смещений (если необходимо).
        std::vector<Scalar> get_derivatives() const override {
            if (BIAS_ACTIVATE) {
                std::vector<Scalar> res(m_dw.size() + m_db.size());

                std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
                std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_dw.size());
                return res;
            } else {
                std::vector<Scalar> res(m_dw.size());
                std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
                return res;
            }

        }

        ///
        /// \return Название слоя.
        std::string layer_type() const override { return "FullyConnected"; }

        ///
        /// \return Название функции активации. Activation::return_type()
        std::string activation_type() const override { return Activation::return_type(); }

        std::string distribution_type() const override { return Distribution::return_type(); }

        /// Формирование словаря с полным описанием слоя для выгрузки/загрузки сети
        /// \param map словарь
        /// \param index индекс слоя в общей сети.
        void fill_meta_info(Meta &map, int index) const override {
            std::string ind = std::to_string(index);

            map.insert(std::make_pair("Layer " + ind, internal::layer_id(layer_type())));
            map.insert(std::make_pair("Activation " + ind, internal::activation_id(activation_type())));
            map.insert(std::make_pair("Distribution " + ind, internal::distribution_id(distribution_type())));
            map.insert(std::make_pair("Bias FC " + ind, BIAS_ACTIVATE));
            map.insert(std::make_pair("in_size " + ind, in_size()));
            map.insert(std::make_pair("out_size " + ind, out_size()));
        }

        friend std::ostream &operator<<(std::ostream &out, FullyConnected &obj) {
            out << std::string(25, ':')
                << "Fully Connected Layer Information"
                << std::string(25, ':') << std::endl << std::endl;

            out << "InOut Info" << std::string(10, '-') << ">" << std::endl;

            out << "In size: " << obj.m_in_size << " " << std::endl
                << "Out size: " << obj.m_out_size << " " << std::endl
                << "Workflow: " << obj.workflow << std::endl << std::endl;

            out << "Activation" << std::string(10, '-') << ">" << obj.activation_type() << std::endl;

            out << "Distribution" << std::string(10, '-') << ">" << obj.distribution_type() << std::endl << std::endl;

            out << "Parameters" << std::string(10, '-') << ">" << std::endl;

            out << "1. Weight's" << std::endl;

            out << obj.m_weight << std::endl;

            if (obj.BIAS_ACTIVATE) {
                out << "2. Bias" << std::endl;

                out << obj.m_bias << std::endl;
            } else {
                out << "2. BIAS BLOCKED" << std::endl;
            }

            return out;
        }
    };
} // namespace xsdnn


#endif //XSDNN_FULLYCONNECTED_H
