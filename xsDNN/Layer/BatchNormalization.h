//
// Created by shuffle on 29.07.22.
//

#ifndef XSDNN_INCLUDE_BATCHNORMALIZATION_H
#define XSDNN_INCLUDE_BATCHNORMALIZATION_H

# include "../Utils/BatchNormUtil.h"
# include "../Utils/Math.h"

/*!
\brief Класс слоя пакетной нормализации (BatchNorm1D)
\image html batchnorm_formula_info.png
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Август 2022 года
*/
template<typename Distribution, typename Activation>
class BatchNorm1D : public Layer
{
private:
    typedef Eigen::VectorXd             Vector;
    typedef Vector::AlignedMapType      AlignedMapVec;

    Matrix      m_z;                    ///< equal      in_hat
    Matrix      m_a;                    ///< equal      gammas * in_hat + betas
    Vector      m_gammas;               ///< gamma для линейного отображения
    Vector      m_betas;                ///< beta для линейного отображения

    Vector      mean_curr;              ///< матожидание для текущего батча
    Vector      var_curr;               ///< дисперсия для текущего батча
    Vector      m_stddev;               ///< стандартное отклонение

    Vector      m_mean;                 ///< скользящее среднее мат ожидания
    Vector      m_var;                  ///< скользящее среднее дисперсии


    Matrix      m_din;
    Vector      m_dg;
    Vector      m_db;

    Scalar      eps;
    Scalar      moment;

    bool        affine_;                 ///< афинное преобразование m_a = m_gammas * m_z + m_betas

public:
    ///
    /// \param in_size длина признакового описания __одного__ объекта
    /// \param tolerance equal eps
    /// \param momentum momentum coefficient
    explicit BatchNorm1D(const int& in_size,
                const bool&         affine = true,
                const Scalar&       tolerance = Scalar(0.0001),
                const Scalar&       momentum  = Scalar(0.35)
                ) : Layer(in_size, in_size, "undefined"),
                    eps(tolerance),
                    moment(momentum),
                    affine_(affine)
                    {}

    void init(const std::vector<Scalar>& params, RNG& rng) override{
        init();

        if (affine_){
            Distribution::set_random_data(m_gammas.data(), m_gammas.size(), rng, params);
            Distribution::set_random_data(m_betas.data(), m_betas.size(), rng, params);
        }
    }

    void init() override{
        if (affine_){
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
    void forward(const Matrix& prev_layer_data) override{
        const long ncols = prev_layer_data.cols();
        m_z.resize(this->m_in_size, ncols);
        m_a.resize(this->m_in_size, ncols);

        Vector mean = (workflow == "train") ? mean_curr : m_mean;
        Vector var  = (workflow == "train") ? var_curr  : m_var;

        if (workflow == "train"){
            internal::math::update_statistics(prev_layer_data, this->m_in_size, mean, var);
        }

        m_stddev = (var.array() + eps).sqrt();

        int index = 0;
        for (auto col : prev_layer_data.colwise()){
            m_z.col(index++) = (col.array() - mean.array()) / m_stddev.array();
        }

        if (affine_){
            m_z = m_z.array().colwise() * m_gammas.array();
            m_z = m_z.array().colwise() + m_betas.array();
        }

        Activation::activate(m_z, m_a);
    }

    ///
    /// \return отнормированные значения нейронов
    const Matrix& output() const override { return m_a; }

    /// Обратный проход по слою
    ///
    /// \image html batchnorm_forward_bacward_pass.png
    ///
    /// Положение предыдущего - следующего слоя равносильно прямому проходу
    /// \param prev_layer_data выходы нейронов предыдущего слоя
    /// \param next_layer_data вектор градиента следующего слоя
    void backprop(const Matrix& prev_layer_data,
                  const Matrix& next_layer_data) override{
        Matrix& dLz = m_z;
        Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);


        assert(m_din.rows() == this->m_in_size);
        assert(m_din.cols() == m_z.cols());

        assert(m_dg.rows() == this->m_in_size);
        assert(m_db.rows() == this->m_in_size);
    }

    ///
    /// \return производная по нейронам
    const Matrix& backprop_data() const override { return m_din; }

    /// Обновление параметров слоя -  векторов _gammas_ и _betas_
    /// \param opt - объект класса Optimizer
    void update(Optimizer& opt) override{
        AlignedMapVec dg(m_dg.data(), m_dg.size());
        AlignedMapVec g(m_gammas.data(), m_gammas.size());
        AlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec b(m_betas.data(), m_betas.size());

        // TODO: обновить через экспоненциальное сглаживание
        opt.update(dg, g);
        opt.update(db, b);
    }

    /// Установить рабочий процесс - тренировка
    void train() override { workflow = "train"; }

    /// Установить рабочий процесс - тестирование
    void eval() override { workflow = "eval"; }

    /// Получить вектор гаммы, беты и среднего мат.ожидания и дисперсии по всем батчам
    /// \return m_gammas, m_betas, m_mean, m_var
    std::vector<Scalar> get_parametrs() const override{
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
    void set_parametrs(const std::vector<Scalar>& param) override{
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
    std::vector<Scalar> get_derivatives() const override{
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

    void fill_meta_info(Meta& map, int index) const override{
        std::string ind = std::to_string(index);

        map.insert(std::make_pair("Layer " + ind, internal::layer_id(layer_type())));
        map.insert(std::make_pair("Distribution " + ind, internal::distribution_id(distribution_type())));
        map.insert(std::make_pair("in_size " + ind, this->in_size()));
        map.insert(std::make_pair("tolerance " + ind, this->eps));
        map.insert(std::make_pair("momentum " + ind, this->moment));
    }

    void set_gamma(Vector gamma){
        m_gammas = gamma;
    }

    void set_beta(Vector beta){
        m_betas = beta;
    }

    void set_stddev(Vector stddev){
        m_stddev = stddev;
    }
};

#endif //XSDNN_INCLUDE_BATCHNORMALIZATION_H
