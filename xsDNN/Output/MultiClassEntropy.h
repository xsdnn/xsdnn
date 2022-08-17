//
// Created by shuffle on 22.06.22.
//

#ifndef XSDNN_INCLUDE_MULTICLASSENTROPY_H
#define XSDNN_INCLUDE_MULTICLASSENTROPY_H

/*!
\brief Класс функционала ошибки для мультиклассовой классификации (MultiClassEntropy)
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Август 2022 года
*/
class CrossEntropyLoss : public Output
{
private:
    Matrix m_din;           ///< Производная по этому слою

public:
    /// Проверка входных данных на соотвествие значений
    /// \param target целевая переменная
    void check_target_data(const Matrix& target) const override
    {
        const long ncols = target.cols();       // eq. n sample's
        const long nrows = target.rows();       // eq. n feature's

        for (long i = 0; i < ncols; i++)
        {
            int n_one = 0;                      // count one in sample - must be 1 one in col

            for (long j = 0; j < nrows; j++)
            {
                if (target(j, i) == Scalar(1)) { n_one++; continue; }

                if (target(j, i) != Scalar(0))
                {
                    throw std::invalid_argument("[class CrossEntropyLoss] Target should only contain zero or one");
                }
            }

            if (n_one != 1)
            {
                throw std::invalid_argument("[class CrossEntropyLoss] Each column of target data should only contain one \"1\"");
            }
        }
    }

    /// \image html multiclassentropy_evaluate.png
    /// \param prev_layer_data последний скрытый слой сети
    /// \param target целевая переменная
    void evaluate(const Matrix& prev_layer_data, const Matrix& target) override
    {
        const long ncols = prev_layer_data.cols();
        const long nrows = prev_layer_data.rows();

        if (ncols != target.cols() || nrows != target.rows())
        {
            throw std::invalid_argument("[class CrossEntropyLoss] Target data have incorrect dimension");
        }

        m_din.resize(nrows, ncols);
        m_din.noalias() = -target.cwiseQuotient(prev_layer_data);
    }

    ///
    /// \return Вектор направления спуска (антиградиент).
    const Matrix& backprop_data() const override
    {
        return m_din;
    }

    /// \image html multiclassentropy_loss.png
    /// \return Ошибку на обучающей выборке
    Scalar loss() const override
    {
        // L = -sum(log(phat) * y)
        // in = phat
        // d(L) / d(in) = -y / phat
        // m_din contains 0 if y = 0, and -1/phat if y = 1
        Scalar res = Scalar(0);
        const int nelem = m_din.size();
        const Scalar* din_data = m_din.data();

        for (int i = 0; i < nelem; i++)
        {
            if (din_data[i] < Scalar(0))
            {
                res += std::log(-din_data[i]);
            }
        }

        return res / m_din.cols();
    }

    ///
    /// \return Тип выходного слоя.
    std::string output_type() const override
    {
        return "MultiClassEntropy";
    }
};

#endif //XSDNN_INCLUDE_MULTICLASSENTROPY_H
