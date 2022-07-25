//
// Created by shuffle on 24.07.22.
//

#ifndef XSDNN_INCLUDE_DROPOUT_H
#define XSDNN_INCLUDE_DROPOUT_H

# include <Eigen/Core>
# include <vector>
# include <stdexcept>

class Dropout : public Layer
{
private:

    Matrix              m_z;                ///< значения входных нейронов
    Matrix              m_a;                ///< значения выходных нейронов
    Matrix              m_din;              ///< вектор градиента по этому слою
    Matrix              mask_;              ///< маска, содержащая распределение Бернулли для отключения нейронов
    Scalar              dropout_rate_;      ///< вероятность отключения
    Scalar              scale_;             ///< коэффицент масштабирования || 1 / (1 - dropout_rate) ||
    internal::bernoulli bernoulli;

public:
    explicit Dropout(const int& in_size, const Scalar& dropout_rate) :
            Layer(in_size, in_size, "undefined"),
            dropout_rate_(dropout_rate),
            scale_(Scalar(1.0) / ( Scalar(1.0) - dropout_rate_)) {}

    void init(const std::vector<Scalar>& params, RNG& rng_) override
    {
        init();
        bernoulli.set_rng(rng_);
    }

    void init() override
    {
        m_z.resize(this->m_in_size, 1);
        m_a.resize(this->m_out_size, 1 );
    }

    void forward(const Matrix& prev_layer_data) override
    {
        const long ncols = prev_layer_data.cols();

        if (workflow == "train")
        {
            bernoulli.set_param(this->dropout_rate_, this->m_out_size);

            mask_.resize(this->m_out_size, ncols);
            m_a.resize(this->m_out_size, ncols);

            for (int i = 0; i < ncols; i++)
            {
                bernoulli(mask_.col(i).data());
            }
            // assert(mask_.size() == prev_layer_data.size());
            m_a = prev_layer_data.cwiseProduct(mask_);
            m_a = m_a * scale_;
            // assert(m_a.size() == prev_layer_data.size());
        }
        else
        {
            m_a = prev_layer_data;
        }
    }

    const Matrix& output() const override { return m_a; }

    void backprop(const Matrix& prev_layer_data,
                  const Matrix& next_layer_data) override
    {
        const long ncols = prev_layer_data.cols();
        m_din.resize(this->m_out_size, ncols);
        m_din = next_layer_data.cwiseProduct(mask_) * scale_;
    }


    const Matrix& backprop_data() const override { return m_din;};

    void update(Optimizer& opt) override {}

    void train() override { workflow = "train"; }

    void eval() override { workflow = "eval"; }

    std::vector<Scalar> get_parametrs() const override { return {}; }

    void set_parametrs(const std::vector<Scalar>& param) override {};

    std::vector<Scalar> get_derivatives() const override { return {}; };

    std::string layer_type() const override { return "Dropout"; };

    std::string activation_type() const override { return "undefined"; };

    std::string distribution_type() const override { return "undefined"; };

    void fill_meta_info(Meta& map, int index) const override {};

};

#endif //XSDNN_INCLUDE_DROPOUT_H
