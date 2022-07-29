//
// Created by shuffle on 29.07.22.
//

#ifndef XSDNN_INCLUDE_BATCHNORMALIZATION_H
#define XSDNN_INCLUDE_BATCHNORMALIZATION_H


// TODO: implement batch normalization graph from https://clck.ru/sQxjL

class BatchNorm1D : public Layer
{
private:
    typedef Eigen::RowVectorXd          RowVector;
    typedef RowVector::AlignedMapType   AlignedMapRowVec;

    Matrix      m_z;
    Matrix      m_a;
    RowVector   m_gammas;
    RowVector   m_betas;

    Matrix      m_din;
    RowVector   m_dg;
    RowVector   m_db;

    Scalar      eps;
    Scalar      moment;

public:
    explicit BatchNorm1D(const int& in_size,
                const Scalar& tolerance = Scalar(0.0001),
                const Scalar& momentum = Scalar(0.01)
                ) : Layer(in_size, in_size, "undefined"), eps(tolerance), moment(momentum) {}

    void init(const std::vector<Scalar>& param, RNG& rng) override {}

    void init() override {}

    void forward(const Matrix& prev_layer_data) override {}

    const Matrix& output() const override { return m_a; }

    void backprop(const Matrix& prev_layer_data,
                  const Matrix& next_layer_data) override {}

    const Matrix& backprop_data() const override { return m_din; }

    void update(Optimizer& opt) override {}

    /// Установить рабочий процесс - тренировка
    void train() override { workflow = "train"; }

    /// Установить рабочий процесс - тестирование
    void eval() override { workflow = "eval"; }

    std::vector<Scalar> get_parametrs() const override { return {}; }

    void set_parametrs(const std::vector<Scalar>& param) override {}

    std::vector<Scalar> get_derivatives() const override { return {}; }

    std::string layer_type() const override { return "BatchNorm1D"; }

    std::string activation_type() const override { return "undefined"; }

    std::string distribution_type() const override { return "undefined"; }

    void fill_meta_info(Meta& map, int index) const override {}
};

#endif //XSDNN_INCLUDE_BATCHNORMALIZATION_H
