#pragma once

# include <Eigen/Core>
# include "Config.h"


class Optimizer
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::AlignedMapType AlignedMapVec;

public:
	virtual ~Optimizer() {}


	///
	/// Сброс информации о текущем оптимайзере
	/// 

	virtual void reset() {};

	///
	/// Собственно метод, отвечающий за обновление весов в сетке
	/// 

	virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};