#pragma once

# include <Eigen/Core>
# include <iostream>
# include "../Callback.h"
# include "../Config.h"
# include "../NeuralNetwork.h"

class CustomClassificationCallback : public Callback
{
private:
	Matrix y_predict_transform(const Matrix& y_pred)
	{
		Matrix y_pred_sign;
		Eigen::MatrixXd ONES = Eigen::MatrixXd::Ones(y_pred.rows(), y_pred.cols());
		y_pred_sign.array() = (y_pred.array() > Scalar(0.5)).select(ONES, Scalar(0));
		return y_pred_sign;
	}
	void accuracy_score(const Scalar* target_data, const Scalar* predict_data, const int& nelem)
	{
		Scalar coincidence = 0;
		for (int i = 0; i < nelem; i++)
		{
			if (target_data[i] == predict_data[i]) { coincidence++; }
		}
		const Scalar accuracy = coincidence / static_cast<Scalar>(nelem);
		std::cout << "---------------------------------------------- accuracy = " << accuracy << std::endl;
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

	void f1_score(const Scalar& precision, const Scalar& recall)
	{
		const Scalar f1 = ((1 + b_koef * b_koef) * precision * recall) / ((b_koef * b_koef) * precision + recall + 0.000001);
		std::cout << "---------------------------------------------- f1-score = " << f1 << std::endl;
	}

public:
	const Scalar b_koef = 1;

	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const Matrix& y)
	{
		const Scalar loss = net->get_output()->loss();
		const Matrix net_output = net->get_last_hidden_layer();
		Matrix y_pred = y_predict_transform(net_output);
		const Scalar* target_data = y.data();
		const Scalar* predict_data = y_pred.data();
		const int nelem = y.size();
		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;
		accuracy_score(target_data, predict_data, nelem);
		const Scalar precision = precision_score(target_data, predict_data, nelem);
		const Scalar recall = recall_score(target_data, predict_data, nelem);
		f1_score(precision, recall);
	}

	void post_trained_batch(const NeuralNetwork* net,
		const Matrix& x,
		const IntegerVector& y)
	{
		const Scalar loss = net->get_output()->loss();

		std::cout << "[Epoch = " << m_epoch_id << ", batch = " << m_batch_id << "] Loss = " << loss << std::endl;
	}
};