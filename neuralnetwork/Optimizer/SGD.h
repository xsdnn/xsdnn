#pragma once

# include <Eigen/Core>
# include "../Config.h"
# include "../Optimizer.h"

class SGD : public Optimizer
{
public:
	Scalar m_lrate;
	Scalar m_decay;

	///
	/// Конструктор по умолчанию, работает всегда. 
	/// Инициализируется стандартными значениями,
	/// которые потом пользователь может изменить.
	/// 

	SGD(const Scalar& lrate = Scalar(0.01), const Scalar& decay = Scalar(0)) :
		m_lrate(lrate), m_decay(decay) {}

	void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
	{
		vec.noalias() -= m_lrate * (dvec + m_decay * vec);
	}
};