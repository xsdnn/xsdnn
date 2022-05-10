#pragma once
# include <cmath>

class CustomCallbackRegressor : public Callback
{
private:
	typedef std::vector<Scalar> Vector;
	int prev_epoch = 0;

	Vector batch_loss;
	bool RESIZE_BATCH_LOSS = true;
	

	inline Scalar mean_batch_loss()
	{
		Scalar sum = 0;
		for (int i = 0; i < m_nbatch; i++)
		{
			sum += batch_loss[i];
		}

		return sum / batch_loss.size();
	}

	inline Scalar median_batch_loss()
	{
		static const int ind = m_nbatch / 2;
		std::nth_element(batch_loss.begin(), batch_loss.begin() + ind, batch_loss.end());
		return Scalar(batch_loss[ind]);
	}

	inline Scalar variance_batch_loss(const Scalar& mean)
	{
		Scalar var = 0;

		for (int i = 0; i < m_nbatch; i++)
		{
			var += (batch_loss[i] - mean) * (batch_loss[i] - mean);
		}

		var /= m_nbatch;
		return var;
	}


	inline void show_result()
	{
		Scalar mean;
		mean = mean_batch_loss();
		std::cout << "Epoch '" << m_epoch_id - 1 << "' is Done! --------------------------- Mean Loss = " << mean << std::endl;

		Scalar median;
		median = median_batch_loss();
		std::cout << "---------------------------------------------- Median Loss = " << median << std::endl;

		Scalar variance;
		variance = variance_batch_loss(mean);
		std::cout << "---------------------------------------------- St. Dev. = " << sqrt(variance) << std::endl;
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

		if ((m_epoch_id - 1 == prev_epoch)) show_result();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;

		if (m_epoch_id == m_nepoch - 1 && m_batch_id == m_nbatch - 1) show_result();
	}

	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const IntegerVector& y)
	{
		if (RESIZE_BATCH_LOSS) batch_loss.resize(m_nbatch); RESIZE_BATCH_LOSS = 0;

		const Scalar loss = net->get_output()->loss();

		batch_loss[m_batch_id] = loss;

		if ((m_epoch_id - 1 == prev_epoch)) show_result();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;

		if (m_epoch_id == m_nepoch - 1 && m_batch_id == m_nbatch - 1) show_result();
	}
};