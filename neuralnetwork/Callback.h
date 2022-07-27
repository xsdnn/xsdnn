#pragma once

# include <Eigen/Core>
# include "Config.h"


class NeuralNetwork;

/*!
	\brief Родительский класс вывода информации на экран
    \author shuffle-true
	\version 1.0
	\date Март 2022 года
	\warning Не следует изменять исходный код этого класса
*/
class Callback
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	int m_nbatch;	// Кол-во батчей
	int m_batch_id; // Текущий батч [0, 1, 2, ..., m_nbatch - 1]
	int m_nepoch;	// Кол-во эпох (кол-во прохождений по всем батчам) 
	int m_epoch_id; // Текущая эпоха [0, 1, 2, ..., m_nepoch - 1]

	Callback() :
		m_nbatch(0), m_batch_id(0), m_nepoch(0), m_epoch_id(0) {}

	virtual ~Callback() = default;

    /*!
     *
     */
	virtual void pre_trained_batch(const NeuralNetwork* net, const Matrix& x,
		const Matrix& y) {}

	virtual void post_trained_batch(const NeuralNetwork* net, const Matrix& x,
		const Matrix& y) {}
};




