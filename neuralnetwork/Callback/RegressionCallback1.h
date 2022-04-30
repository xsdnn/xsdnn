#pragma once


class RegressionCallback : public Callback
{
private:
	typedef std::vector<Scalar> Vector;
	int prev_epoch = 0;

	Vector batch_loss;
	bool RESIZE_BATCH_LOSS = true;
	

	Scalar mean_batch_loss()
	{
		Scalar sum = 0;
		for (int i = 0; i < m_nbatch; i++)
		{
			sum += batch_loss[i];
		}

		return sum / batch_loss.size();
	}

	inline void show_mean_result()
	{
		Scalar mean;
		mean = mean_batch_loss();
		std::cout << "Epoch '" << m_epoch_id - 1 << "' is Done! --------------------------- Mean Loss = " << mean << std::endl;
		prev_epoch++;
	}

public:
	
	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const Matrix& y)
	{

		// выделяем память под вектор лосса по батчам один раз за все обучение - в самом начале.
		if (RESIZE_BATCH_LOSS) batch_loss.resize(m_nbatch); RESIZE_BATCH_LOSS = 0;

		// считаем лосс
		const Scalar loss = net->get_output()->loss();

		// перезаписываем лосс для каждого из батчей
		batch_loss[m_batch_id] = loss;

		if ((m_epoch_id - 1 == prev_epoch)) show_mean_result();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;

		if (m_epoch_id == m_nepoch - 1 && m_batch_id == m_nbatch - 1) show_mean_result();
	}

	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const IntegerVector& y)
	{
		if (RESIZE_BATCH_LOSS) batch_loss.resize(m_nbatch); RESIZE_BATCH_LOSS = 0;

		const Scalar loss = net->get_output()->loss();

		batch_loss[m_batch_id] = loss;

		if ((m_epoch_id - 1 == prev_epoch)) show_mean_result();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;

		if (m_epoch_id == m_nepoch - 1 && m_batch_id == m_nbatch - 1) show_mean_result();
	}
};