#pragma once

# include <Eigen/Core>
# include "../Config.h"
# include "../Optimizer.h"

/*!
 * Класс управления стохастическим градиентным спуском.
 */
class SGD : public Optimizer
{
private:
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Vector;
    std::map<const Scalar*, Vector> momentum_history;

public:
	Scalar m_lrate;     ///< длина шага.
	Scalar m_decay;     ///< коэффицент линейной зависимости с производной.
	Scalar m_momentum;  ///< импульс спуска
	Scalar m_dampening; ///< сглаживание момента
    bool   m_nesterov;  ///< активация момента Нестерова

    ///
    /// \param lrate длина шага.
    /// \param decay weight_decay L2-penalty.
    /// \param momentum momentum factor (default: 0)
    /// \param dampening dampening for momentum (default: 0)
    /// \param nesterov enables Nesterov momentum (default: false)
	explicit SGD(const Scalar& lrate = Scalar(0.01), const Scalar& decay = Scalar(0),
        const Scalar& momentum = Scalar(0.0), const bool& nesterov = false, const Scalar& dampening = Scalar(0.0)) :
		m_lrate(lrate), m_decay(decay), m_momentum(momentum), m_dampening(dampening), m_nesterov(nesterov) {}

    /// Сброс истории момента
    void reset() override
    {
        momentum_history.clear();
    }

    /// \image html sgd_implementation.jpg
    /// \param grad вектор производной (например веса или смещения)
    /// \param theta вектор значений (например веса или смещения)
	void update(AlignedMapVec& grad, AlignedMapVec& theta) override
	{
        if (m_decay != 0) grad += m_decay * theta;

        if (m_momentum != 0)
        {
            Vector& beta = momentum_history[grad.data()];

            if (beta.size() != 0)
            {
                beta = m_momentum * beta + (1 - m_dampening) * grad.array();
            }
            else
            {
                beta = grad;
            }

            if (m_nesterov)
            {
                grad.array() = grad.array() + m_momentum * beta;
            }
            else
            {
                grad.array() = beta;
            }
        }
        theta.noalias() -= m_lrate * grad;
	}
};