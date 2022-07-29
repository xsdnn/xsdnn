#pragma once


/*!
\brief Класс отображения информации при обучении (VerboseCallback)
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Март 2022 года
*/
class VerboseCallback : public Callback
{
public:
    /// Вывод информации в консоль. [ Epoch = ..., batch = ... ] loss = ...
    /// \param net объекта класса сетки.
    /// \param x обучающая выборка.
    /// \param y таргет.
	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const Matrix& y) override
	{
		const Scalar loss = net->get_output()->loss();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;
	}
};