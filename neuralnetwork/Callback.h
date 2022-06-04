#pragma once

# include <Eigen/Core>
# include "Config.h"


class NeuralNetwork;

/*
Базовый класс для отправки пользователю сообщений о тренировке сетки.
Будет предусмотрена возможность вывода на экран основных тренировочных характеристик сетки.
*/

class Callback
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::RowVectorXi IntegerVector;

public:
	int m_nbatch;	// Кол-во батчей
	int m_batch_id; // Текущий батч [0, 1, 2, ..., m_nbatch - 1]
	int m_nepoch;	// Кол-во эпох (кол-во прохождений по всем батчам) 
	int m_epoch_id; // Текущая эпоха [0, 1, 2, ..., m_nepoch - 1]

	Callback() :
		m_nbatch(0), m_batch_id(0), m_nepoch(0), m_epoch_id(0) {}

	virtual ~Callback() = default;

	// Перед тренировкой батча
	virtual void pre_trained_batch(const NeuralNetwork* net, const Matrix& x,
		const Matrix& y) {}

	virtual void pre_trained_batch(const NeuralNetwork* net, const Matrix& x,
		const IntegerVector& y) {}

	// После тренировки батча
	virtual void post_trained_batch(const NeuralNetwork* net, const Matrix& x,
		const Matrix& y) {}

	virtual void post_trained_batch(const NeuralNetwork* net, const Matrix& x,
		const IntegerVector& y) {}
};




