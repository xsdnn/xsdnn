#pragma once

# include <Eigen/Core>
# include "../Config.h"
# include "../Optimizer.h"

/*!
 * Класс управления стохастическим градиентным спуском.
 */
class SGD : public Optimizer
{
public:
	Scalar m_lrate; ///< длина шага.
	Scalar m_decay; ///< коэффицент линейной зависимости с производной.

    ///
    /// \param lrate длина шага.
    /// \param decay коэффицент линейной зависимости с производной.
	SGD(const Scalar& lrate = Scalar(0.01), const Scalar& decay = Scalar(0)) :
		m_lrate(lrate), m_decay(decay) {}

    /// Обновление весов слоя по формуле ->
    ///
    /// w_0 -= m_lrate * (dvec + m_decay * vec);
    /// \param dvec вектор производной (например веса или смещения)
    /// \param vec вектор значений (например веса или смещения)
	void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
	{
		vec.noalias() -= m_lrate * (dvec + m_decay * vec);
	}
};