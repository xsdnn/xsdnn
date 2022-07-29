//
// Created by shuffle on 29.07.22.
//

#ifndef XSDNN_INCLUDE_BATCHNORMALIZATION_H
#define XSDNN_INCLUDE_BATCHNORMALIZATION_H

# include "../Utils/BatchNormUtil.h"

// TODO: implement batch normalization graph from https://clck.ru/sQxjL

template<typename Distribution>
class BatchNorm1D : public Layer
{
private:
    typedef Eigen::VectorXd             Vector;
    typedef Vector::AlignedMapType      AlignedMapRowVec;

    Matrix      m_z;                    ///< equal      in_hat
    Matrix      m_a;                    ///< equal      gammas * in_hat + betas
    Vector      m_gammas;               ///< gamma для линейного отображения
    Vector      m_betas;                ///< beta для линейного отображения

    Vector      m_mean;                 ///< скользящее среднее мат ожидания
    Vector      m_var;                  ///< скользящее среднее дисперсии

    Matrix      m_din;
    Vector      m_dg;
    Vector      m_db;

    Scalar      eps;
    Scalar      moment;
    Scalar      lmbd;                   ///< коэфф при скользящем среднем статистик

public:
    ///
    /// \param in_size длина признакового описания __одного__ объекта
    /// \param tolerance equal eps
    /// \param momentum momentum coefficient
    explicit BatchNorm1D(const int& in_size,
                const Scalar&       tolerance = Scalar(0.0001),
                const Scalar&       momentum  = Scalar(0.01),
                const Scalar&       lambda    = Scalar(0.35)
                ) : Layer(in_size, in_size, "undefined"),
                    eps(tolerance),
                    moment(momentum),
                    lmbd(lambda)
                    {}

    void init(const std::vector<Scalar>& params, RNG& rng) override
    {
        init();

        Distribution::set_random_data(m_gammas.data(), m_gammas.size(), rng, params);
        Distribution::set_random_data(m_betas.data(), m_betas.size(), rng, params);

        assert(m_gammas.size() == this->m_in_size);
        assert(m_betas.size() == this->m_in_size);

//        std::cout << m_gammas << std::endl;
//        std::cout << m_betas << std::endl;
    }

    void init() override
    {
        m_gammas.resize(this->m_in_size);
        m_betas.resize(this->m_in_size);

        m_mean.resize(this->m_in_size);
        m_var.resize(this->m_in_size);

        m_mean.setZero();
        m_var.setZero();

        m_db.resize(this->m_in_size);
        m_dg.resize(this->m_in_size);
    }

    void forward(const Matrix& prev_layer_data) override
    {
        const long ncols = prev_layer_data.cols();
        m_z.resize(this->m_in_size, ncols);
        m_a.resize(this->m_in_size, ncols);

        if (workflow == "train")
        {
            internal::compute_statistic_graph1D_train(prev_layer_data, m_z, m_mean, m_var, eps, lmbd);
            m_a = m_z.array().colwise() * m_gammas.array();
            m_a = m_a.array().colwise() + m_betas.array();
            assert(m_a.rows() == this->m_in_size);
            assert(m_a.cols() == ncols);
        }
        else
        {
            internal::compute_statistic_graph1D_eval(m_z, m_mean, m_var, eps);
            m_a = m_z.array().colwise() * m_gammas.array();
            m_a = m_a.array().colwise() + m_betas.array();
            assert(m_a.rows() == this->m_in_size);
            assert(m_a.cols() == ncols);
        }
    }

    const Matrix& output() const override { return m_a; }

    void backprop(const Matrix& prev_layer_data,
                  const Matrix& next_layer_data) override {}

    const Matrix& backprop_data() const override { return m_din; }

    void update(Optimizer& opt) override {}

    /// Установить рабочий процесс - тренировка
    void train() override { workflow = "train"; }

    /// Установить рабочий процесс - тестирование
    void eval() override { workflow = "eval"; }

    /// Получить вектор гаммы, беты и среднего мат.ожидания и дисперсии по всем батчам
    /// \return m_gammas, m_betas, m_mean, m_var
    std::vector<Scalar> get_parametrs() const override
    {
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

    /// Установить параметры слоя - гаммы, беты и среднего мат.ожидания и дисперсии по всем батчам
    /// \param param вектор всех параметров слоя
    void set_parametrs(const std::vector<Scalar>& param) override
    {
        if (param.size() != m_gammas.size() + m_betas.size() + m_mean.size() + m_var.size())
        {
            throw std::invalid_argument("[class BatchNorm1D]: Parameter size does not match. Check parameter size!");
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
    }

    /// Получить производные по значениям нейрона, по гамме и бете
    /// \return m_din, m_dg, m_db
    std::vector<Scalar> get_derivatives() const override
    {
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

    std::string layer_type() const override { return "BatchNorm1D"; }

    std::string activation_type() const override { return "undefined"; }

    std::string distribution_type() const override { return Distribution::return_type(); }

    void fill_meta_info(Meta& map, int index) const override
    {
        std::string ind = std::to_string(index);

        map.insert(std::make_pair("Layer" + ind, internal::layer_id(layer_type())));
        map.insert(std::make_pair("Distribution " + ind, internal::distribution_id(distribution_type())));
        map.insert(std::make_pair("in_size " + ind, this->in_size()));
        map.insert(std::make_pair("tolerance " + ind, this->eps));
        // TODO: подумать что еще можно сохранять
    }
};

#endif //XSDNN_INCLUDE_BATCHNORMALIZATION_H
