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

    Matrix      m_z;                ///< значения входных нейронов
    Matrix      m_a;                ///< значения выходных нейронов
    Matrix      mask_;              ///< маска, содержащая распределение Бернулли для отключения нейронов
    Scalar      dropout_rate_;      ///< вероятность отключения
    Scalar      scale_;             ///< коэффицент масштабирования || 1 / (1 - dropout_rate) ||

public:
    explicit Dropout(const int& in_size, const Scalar& dropout_rate) :
            Layer(in_size, in_size, "undefined"),
            dropout_rate_(dropout_rate),
            scale_(Scalar(1.0) / ( Scalar(1.0) - dropout_rate_)) {}

    void init(const std::vector<Scalar>& params, RNG& rng_) override
    {
        // TODO: создать объект класса Бернулли для дальнейшего использования при прямом и обратном проходе по слою
        init();
    }

    void init() override
    {
        m_z.resize(this->m_in_size, 1);
        m_a.resize(this->m_out_size, 1 );
        mask_.resize(this->m_out_size, 1);
    }

    void forward(const Matrix& prev_layer_data) override
    {
        const long ncols = prev_layer_data.cols();

        mask_.resize(this->m_out_size, ncols);

        for (int i = 0; i < ncols; i++)
        {

        }
    }
};

#endif //XSDNN_INCLUDE_DROPOUT_H
