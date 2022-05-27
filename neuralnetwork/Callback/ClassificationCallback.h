#pragma once

# include <Eigen/Core>
# include <iostream>
# include "../Callback.h"
# include "../Config.h"
# include "../NeuralNetwork.h"

class CustomClassificationCallback : public Callback
{
private:
	typedef std::vector<Scalar> Vector;
	int prev_epoch = 0;

	Vector batch_loss;
	Vector accuracy_score_vector;
	Vector precision_score_vector;
	Vector recall_score_vector;
	Vector f1_score_vector;
	bool RESIZE_VECTOR = true;

	Matrix y_predict_transform(const Matrix& y_pred)
	{
		Matrix y_pred_sign;
		Eigen::MatrixXd ONES = Eigen::MatrixXd::Ones(y_pred.rows(), y_pred.cols());
		y_pred_sign.array() = (y_pred.array() > Scalar(0.5)).select(ONES, Scalar(0));
		return y_pred_sign;
	}
	const Scalar accuracy_score(const Scalar* target_data, const Scalar* predict_data, const int& nelem)
	{
		Scalar coincidence = 0;
		for (int i = 0; i < nelem; i++)
		{
			if (target_data[i] == predict_data[i]) { coincidence++; }
		}
		const Scalar accuracy = coincidence / static_cast<Scalar>(nelem);
		std::cout << "---------------------------------------------- accuracy = " << accuracy << std::endl;
		return accuracy;
	}

	const Scalar precision_score(const Scalar* target_data, const Scalar* predict_data, const int& nelem)
	{
		int TP = 0;
		int FP = 0;

		for (int i = 0; i < nelem; i++)
		{
			if ((target_data[i] == predict_data[i]) && (target_data[i] == 1)) { TP++; }

			if ((target_data[i] != predict_data[i]) && (predict_data[i] == 1)) { FP++; }
		}

		const Scalar precision = TP / (TP + FP + 0.000001);
		std::cout << "---------------------------------------------- precision = " << precision << std::endl;
		return precision;
	}

	const Scalar recall_score(const Scalar* target_data, const Scalar* predict_data, const int& nelem)
	{
		int TP = 0;
		int FN = 0;

		for (int i = 0; i < nelem; i++)
		{
			if ((target_data[i] == predict_data[i]) && (target_data[i] == 1)) { TP++; }

			if ((target_data[i] != predict_data[i]) && (predict_data[i] == 0)) { FN++; }
		}

		const Scalar recall = TP / (TP + FN + 0.000001);
		std::cout << "---------------------------------------------- recall = " << recall << std::endl;
		return recall;
	}

	const Scalar f1_score(const Scalar& precision, const Scalar& recall)
	{
		const Scalar f1 = ((1 + b_koef * b_koef) * precision * recall) / ((b_koef * b_koef) * precision + recall + 0.000001);
		std::cout << "---------------------------------------------- f1-score = " << f1 << std::endl;
		return f1;
	}


	inline Scalar mean_vector(Vector& vector)
	{
		Scalar sum = 0;
		for (int i = 0; i < m_nbatch; i++)
		{
			sum += vector[i];
		}
		return sum / m_nbatch;
	}

	inline void show_epoch_result()
	{
		Scalar mean;
		Scalar median;
		Scalar variance;

		mean = mean_vector(batch_loss);
		std::cout << "------------------------------------------------------------------ mean loss = " << mean << std::endl;


		mean = mean_vector(accuracy_score_vector);
		std::cout << "------------------------------------------------------------------ mean accuracy = " << mean << std::endl;


		mean = mean_vector(precision_score_vector);
		std::cout << "------------------------------------------------------------------ mean precision = " << mean << std::endl;


		mean = mean_vector(recall_score_vector);
		std::cout << "------------------------------------------------------------------ mean recall = " << mean << std::endl;


		mean = mean_vector(f1_score_vector);
		std::cout << "------------------------------------------------------------------ mean f1 = " << mean << std::endl;
		prev_epoch++;
	}


public:
	const Scalar b_koef = 1;

	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const Matrix& y)
	{
		// resize on first iteration only
		if (RESIZE_VECTOR) 
		{
			batch_loss.resize(m_nbatch);
			accuracy_score_vector.resize(m_nbatch);
			precision_score_vector.resize(m_nbatch);
			recall_score_vector.resize(m_nbatch);
			f1_score_vector.resize(m_nbatch);
			RESIZE_VECTOR = 0;
		}


		const Scalar loss = net->get_output()->loss();
		const Matrix net_output = net->get_last_hidden_layer();
		Matrix y_pred = y_predict_transform(net_output);
		const Scalar* target_data = y.data();
		const Scalar* predict_data = y_pred.data();
		const int nelem = y.size();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;

		const Scalar accuracy = accuracy_score(target_data, predict_data, nelem);
		const Scalar precision = precision_score(target_data, predict_data, nelem);
		const Scalar recall = recall_score(target_data, predict_data, nelem);
		const Scalar f1 = f1_score(precision, recall);

		batch_loss[m_batch_id] = loss;
		accuracy_score_vector[m_batch_id] = accuracy;
		precision_score_vector[m_batch_id] = precision;
		recall_score_vector[m_batch_id] = recall;
		f1_score_vector[m_batch_id] = f1;

		if ((m_epoch_id - 1 == prev_epoch)) show_epoch_result();
	}
};