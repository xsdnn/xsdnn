//
// Created by shuffle on 29.07.22.
//

#ifndef XSDNN_INCLUDE_BATCHNORMUTIL_H
#define XSDNN_INCLUDE_BATCHNORMUTIL_H

# include <Eigen/Core>
# include "../Config.h"

namespace internal{
    /// Получение нормированного входа по вычислительному графу для слоя BatchNorm1D
    /// \param in исходные входные данные
    /// \param in_hat отнормированные входные данные
    inline void compute_statistic_graph1D_train(
                const Matrix& in,
                Matrix& in_hat,
                Eigen::VectorXd& m_mean,
                Eigen::VectorXd& m_var,
                const Scalar& eps,
                const Scalar& lbd
            )
    {
        const long ncols  = in.cols();
        const long nrows  = in.rows();

        Eigen::VectorXd mu = ( Scalar(1.0) / static_cast<Scalar>(ncols) ) * in.rowwise().sum();

        assert(mu.cols() == m_mean.cols());
        assert(mu.rows() == m_mean.rows());

        // вычисление скользящего среднего мат ожидания
        m_mean = lbd * m_mean + (1 - lbd) * mu;

        // compute xmu
        in_hat = in.colwise() - mu;

        Eigen::VectorXd var
                            =
                                ( Scalar(1.0) / static_cast<Scalar>(ncols) ) * in_hat.array().square().rowwise().sum();

        assert(var.cols() == m_var.cols());
        assert(var.rows() == m_var.rows());

        // вычисление скользящего среднего дисперсии
        m_var = lbd * m_var + (1 - lbd) * var;

        Eigen::VectorXd denom_var = Scalar(1.0) / (var.array() + eps).array().sqrt();

        // final
        in_hat = in_hat.array().colwise() * denom_var.array();
    }

    inline void compute_statistic_graph1D_eval(
            Matrix& in_hat,
            Eigen::VectorXd& m_mean,
            Eigen::VectorXd& m_var,
            const Scalar& eps
    )
    {
        // compute xmu
        in_hat = in_hat.colwise() - m_mean;

        // compute denominator mean var
        Eigen::VectorXd  denom_var = Scalar(1.0) / (m_var.array() + eps).array().sqrt();

        // final
        in_hat = in_hat.array().colwise() * denom_var.array();
    }

}

#endif //XSDNN_INCLUDE_BATCHNORMUTIL_H
