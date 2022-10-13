//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_INCLUDE_BATCHNORMALIZATION_H
#define XSDNN_INCLUDE_BATCHNORMALIZATION_H

# include "../Core/BATCHNORM/BatchNorm1DCore.h"
# include "../Utils/Except.h"

namespace xsdnn {
    /*!
    \brief Класс слоя пакетной нормализации (BatchNorm1D)
    \image html batchnorm_formula_info.png
    \author __[shuffle-true](https://github.com/shuffle-true)__
    \version 0.0
    \date Август 2022 года
    */
    template<typename Distribution, typename Activation>
    class BatchNorm1D : public Layer {
    public:
        ///
        /// \param in_size длина признакового описания __одного__ объекта
        /// \param tolerance equal eps
        /// \param momentum momentum coefficient
        explicit BatchNorm1D(const int &in_size,
                             const bool &affine = true,
                             const Scalar &tolerance = Scalar(0.0001),
                             const Scalar &momentum = Scalar(0.35)
        ) : Layer(in_size, in_size, "undefined"),
            eps(tolerance),
            moment(momentum),
            affine_(affine) {}

        void init(const std::vector<Scalar> &params, RNG &rng) override {
            init();

            if (affine_) {
                Distribution::set_random_data(m_gammas.data(), m_gammas.size(), rng, params);
                Distribution::set_random_data(m_betas.data(), m_betas.size(), rng, params);
            }
        }

        void init() override {
            if (affine_) {
                m_gammas.resize(this->m_in_size);
                m_betas.resize(this->m_in_size);
                m_dg.resize(this->m_in_size);
                m_db.resize(this->m_in_size);
            }

            mean_curr.resize(this->m_in_size);
            var_curr.resize(this->m_in_size);
            m_stddev.resize(this->m_in_size);

            m_mean.resize(this->m_in_size);
            m_var.resize(this->m_in_size);
            m_mean.setZero();
            m_var.setZero();
        }

        /// Прямой проход по слою
        ///
        /// \image html batchnorm_forward_bacward_pass.png
        ///
        /// \param prev_layer_data значения нейронов предыдущего слоя
        void forward(const xsTypes::Matrix &prev_layer_data) override {

            internal::bn1d::computeForward<Activation>(
                    prev_layer_data, m_z, m_a,
                    mean_curr, var_curr,
                    m_mean, m_var, m_stddev,
                    m_gammas, m_betas,
                    workflow, eps, affine_, this->m_in_size
            );

        }

        ///
        /// \return отнормированные значения нейронов
        const xsTypes::Matrix &output() const override { return m_a; }

        /// Обратный проход по слою
        ///
        /// \image html batchnorm_forward_bacward_pass.png
        ///
        /// Положение предыдущего - следующего слоя равносильно прямому проходу
        /// \param prev_layer_data выходы нейронов предыдущего слоя
        /// \param next_layer_data вектор градиента следующего слоя
        void backprop(const xsTypes::Matrix &prev_layer_data,
                      const xsTypes::Matrix &next_layer_backprop_data) override {

            internal::bn1d::computeBackward<Activation>(
                    prev_layer_data, next_layer_backprop_data,
                    m_z, m_a,
                    m_din, m_dg, m_db,
                    m_stddev, affine_, this->m_in_size
            );

        }

        ///
        /// \return производная по нейронам
        const xsTypes::Matrix &backprop_data() const override { return m_din; }

        /// Обновление параметров слоя -  векторов _gammas_ и _betas_
        /// \param opt - объект класса Optimizer
        void update(Optimizer &opt) override {
            m_mean = moment * m_mean + (1 - moment) * mean_curr;
            m_var = moment * m_var + (1 - moment) * var_curr;

            if (affine_) {
                xsTypes::AlignedMapVec dg(m_dg.data(), m_dg.size());
                xsTypes::AlignedMapVec g(m_gammas.data(), m_gammas.size());
                xsTypes::AlignedMapVec db(m_db.data(), m_db.size());
                xsTypes::AlignedMapVec b(m_betas.data(), m_betas.size());

                opt.update(dg, g);
                opt.update(db, b);
            }
        }

        /// Установить рабочий процесс - тренировка
        void train() override { workflow = "train"; }

        /// Установить рабочий процесс - тестирование
        void eval() override { workflow = "eval"; }

        /// Получить вектор гаммы, беты и среднего мат.ожидания и дисперсии по всем батчам
        /// \return m_gammas, m_betas, m_mean, m_var
        std::vector<Scalar> get_parametrs() const override {
            if (affine_) {
                std::vector<Scalar> res(m_gammas.size() + m_betas.size() + m_mean.size() + m_var.size());

                std::copy(
                        m_gammas.data(),
                        m_gammas.data() + m_gammas.size(),
                        res.begin()
                );

                std::copy(
                        m_betas.data(),
                        m_betas.data() + m_betas.size(),
                        res.begin() + m_gammas.size()
                );

                std::copy(
                        m_mean.data(),
                        m_mean.data() + m_mean.size(),
                        res.begin() + m_gammas.size() + m_betas.size()
                );

                std::copy(
                        m_var.data(),
                        m_var.data() + m_var.size(),
                        res.begin() + m_gammas.size() + m_betas.size() + m_mean.size()
                );
                return res;
            }

            std::vector<Scalar> res(m_mean.size() + m_var.size());

            std::copy(
                    m_mean.data(),
                    m_mean.data() + m_mean.size(),
                    res.begin()
            );

            std::copy(
                    m_var.data(),
                    m_var.data() + m_var.size(),
                    res.begin() + m_mean.size()
            );

            return res;
        }

        /// Установить параметры слоя - гаммы, беты и среднего мат.ожидания и дисперсии по всем батчам
        /// \param param вектор всех параметров слоя
        void set_parametrs(const std::vector<Scalar> &param) override {
            if (affine_) {
                if (param.size() != m_gammas.size() + m_betas.size() + m_mean.size() + m_var.size()) {
                    throw internal::except::xs_error(
                            "[class BatchNorm1D]: Parameter size does not match. Check parameter size!");
                }

                std::copy(
                        param.begin(),
                        param.begin() + m_gammas.size(),
                        m_gammas.data()
                );

                std::copy(
                        param.begin() + m_gammas.size(),
                        param.begin() + m_gammas.size() + m_betas.size(),
                        m_betas.data()
                );

                std::copy(
                        param.begin() + m_gammas.size() + m_betas.size(),
                        param.begin() + m_gammas.size() + m_betas.size() + m_mean.size(),
                        m_mean.data()
                );

                std::copy(
                        param.begin() + m_gammas.size() + m_betas.size() + m_mean.size(),
                        param.begin() + m_gammas.size() + m_betas.size() + m_mean.size() + m_var.size(), // param.end()
                        m_var.data()
                );
            } else {
                if (param.size() != m_mean.size() + m_var.size()) {
                    throw internal::except::xs_error(
                            "[class BatchNorm1D]: Parameter size does not match. Check parameter size!");
                }

                std::copy(
                        param.begin(),
                        param.begin() + m_mean.size(),
                        m_mean.data()
                );

                std::copy(
                        param.begin() + m_mean.size(),
                        param.end(),
                        m_var.data()
                );
            }
        }

        /// Получить производные по значениям нейрона, по гамме и бете
        /// \return m_din, m_dg, m_db
        std::vector<Scalar> get_derivatives() const override {
            if (affine_) {
                std::vector<Scalar> res(m_din.size() + m_dg.size() + m_db.size());

                std::copy(
                        m_din.data(),
                        m_din.data() + m_din.size(),
                        res.begin()
                );

                std::copy(
                        m_dg.data(),
                        m_dg.data() + m_dg.size(),
                        res.begin() + m_din.size()
                );

                std::copy(
                        m_db.data(),
                        m_db.data() + m_db.size(),
                        res.begin() + m_din.size() + m_dg.size()
                );
                return res;
            }

            std::vector<Scalar> res(m_din.size());

            std::copy(
                    m_din.data(),
                    m_din.data() + m_din.size(),
                    res.begin()
            );
            return res;
        }

        std::string layer_type() const override { return "BatchNorm1D"; }

        std::string activation_type() const override { return Activation::return_type(); }

        std::string distribution_type() const override { return Distribution::return_type(); }

        void fill_meta_info(Meta &map, int index) const override {
            std::string ind = std::to_string(index);

            map.insert(std::make_pair("Layer " + ind, internal::layer_id(layer_type())));
            map.insert(std::make_pair("Distribution " + ind, internal::distribution_id(distribution_type())));
            map.insert(std::make_pair("Affine " + ind, this->affine_));
            map.insert(std::make_pair("in_size " + ind, this->in_size()));
            map.insert(std::make_pair("tolerance " + ind, this->eps));
            map.insert(std::make_pair("momentum " + ind, this->moment));
        }

        void set_gamma(xsTypes::Vector gamma) {
            m_gammas = gamma;
        }

        void set_beta(xsTypes::Vector beta) {
            m_betas = beta;
        }

        void set_stddev(xsTypes::Vector stddev) {
            m_stddev = stddev;
        }

        void set_m_z(xsTypes::Matrix m_z_) {
            m_z = m_z_;
        }

    private:
        xsTypes::Matrix m_z;                    ///< equal      in_hat
        xsTypes::Matrix m_a;                    ///< equal      gammas * in_hat + betas
        xsTypes::Vector m_gammas;               ///< gamma для линейного отображения
        xsTypes::Vector m_betas;                ///< beta для линейного отображения

        xsTypes::Vector mean_curr;              ///< матожидание для текущего батча
        xsTypes::Vector var_curr;               ///< дисперсия для текущего батча
        xsTypes::Vector m_stddev;               ///< стандартное отклонение

        xsTypes::Vector m_mean;                 ///< скользящее среднее мат ожидания
        xsTypes::Vector m_var;                  ///< скользящее среднее дисперсии

        xsTypes::Matrix m_din;
        xsTypes::Vector m_dg;
        xsTypes::Vector m_db;

        Scalar eps;
        Scalar moment;

        bool affine_;                 ///< афинное преобразование m_a = m_gammas * m_z + m_betas

    };
} // namespace xsdnn

#endif //XSDNN_INCLUDE_BATCHNORMALIZATION_H
