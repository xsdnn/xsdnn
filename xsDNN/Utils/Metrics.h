//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_METRICS_H
#define XSDNN_METRICS_H

namespace metrics {
    void MSE_calculate(Matrix &target_data, Matrix &net_output, bool sqrt = false) {

        if ((target_data.cols() != net_output.cols()) || (target_data.rows() != net_output.rows())) {
            throw std::invalid_argument("[void MSE_calculate]: Input data have incorrect dimension");
        }

        Scalar MSE = (target_data - net_output).squaredNorm() / static_cast<Scalar>(target_data.cols());
        if (sqrt) {
            std::cout << "RMSE Error = " << std::sqrt(MSE) << std::endl;
            return;
        }
        std::cout << "MSE Error = " << MSE << std::endl;
    }

    void MAE_calculate(Matrix &target_data, Matrix &net_output) {

        if ((target_data.cols() != net_output.cols()) || (target_data.rows() != net_output.rows())) {
            throw std::invalid_argument("[void MAE_calculate]: Input data have incorrect dimension");
        }

        Scalar MAE = (target_data - net_output).array().cwiseAbs().sum() / static_cast<Scalar>(target_data.cols());
        std::cout << "MAE Error = " << MAE << std::endl;
    }

    void R_calculate(Matrix &target_data, Matrix &net_output) {

        if ((target_data.cols() != net_output.cols()) || (target_data.rows() != net_output.rows())) {
            throw std::invalid_argument("[void R_calculate]: Input data have incorrect dimension");
        }

        Scalar MSE = (target_data - net_output).squaredNorm();
        Matrix target_data_mean = target_data.rowwise().mean();

        for (int i = 0; i < target_data_mean.size(); i++) {
            target_data.row(i).array() -= static_cast<Scalar>(target_data_mean(i, 0));
        }
        Scalar r_coef = target_data.squaredNorm();
        std::cout << "R^2 = " << (1 - (MSE / r_coef)) << std::endl;
    }

    void MAPE_calculate(Matrix &target_data, Matrix &net_output) {

        if ((target_data.cols() != net_output.cols()) || (target_data.rows() != net_output.rows())) {
            throw std::invalid_argument("[void MAPE_calculate]: Input data have incorrect dimension");
        }

        Scalar MAPE = ((target_data - net_output).array().cwiseAbs().sum() / target_data.cwiseAbs().array().sum()) /
                      static_cast<Scalar>(target_data.cols());
        std::cout << "MAPE Error = " << MAPE * 100.0 << "%" << std::endl;
    }

    void regressor_metrics_calculate(Matrix &target_data, Matrix &net_output) {
        MSE_calculate(target_data, net_output);
        MAE_calculate(target_data, net_output);
        R_calculate(target_data, net_output);
        MAPE_calculate(target_data, net_output);
    }
}


namespace internal {
    Matrix static build_confusion_matrix(const Scalar *real, const Scalar *predict, const long &sample_size) {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        for (int i = 0; i < sample_size; i++) {
            if ((real[i] == 0) && (predict[i] == 0)) tn++;
            else if ((real[i] == 1) && (predict[i] == 1)) tp++;
            else if ((real[i] == 0) && (predict[i] == 1)) fp++;
            else if ((real[i] == 1) && (predict[i] == 0)) fn++;
        }

        // after calculate value we can build confusion matrix
        Matrix CM(2, 2);
        CM << tn, fn,
                fp, tp;
        return CM;
    }
}

namespace metrics {
    /// Подсчет основных метрик классификации. Строится confusion matrix и по правилам вычисляются нужные метрики
    /// \param y_true матрица ответов
    /// \param y_pred матрица предсказаний, выданная сеткой
    void accuracy_precision_recall_f1_calculate(const Matrix &y_true, const Matrix &y_pred) {

        if ((y_true.cols() != y_pred.cols()) || (y_true.rows() != y_pred.rows())) {
            throw std::invalid_argument(
                    "[void accuracy_precision_recall_f1_calculate]: Input data have incorrect dimension");
        }

        const long sample_size = y_true.size();
        const Scalar *real = y_true.data();
        const Scalar *predict = y_pred.data();
        Matrix CM = internal::build_confusion_matrix(real, predict, sample_size);

        Scalar accuracy = CM.diagonal().sum() / CM.sum();
        Scalar precision = ((CM.diagonal().array()) / (CM.rowwise().sum().array())).mean();
        Scalar recall = ((CM.diagonal().array().transpose()) / (CM.colwise().sum().array())).mean();
        Scalar f1_score = 2 * (precision * recall) / (precision + recall);

        std::cout << "accuracy = " << accuracy << std::endl;
        std::cout << "precision = " << precision << std::endl;
        std::cout << "recall = " << recall << std::endl;
        std::cout << "f1_score = " << f1_score << std::endl;
    }
}


#endif //XSDNN_METRICS_H
