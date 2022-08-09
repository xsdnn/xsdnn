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
                const Matrix&       in,
                Matrix&             in_hat,

                Eigen::VectorXd&    m_mean,
                Eigen::VectorXd&    m_var,

                Matrix&             inmu,
                Eigen::VectorXd&    invvar,

                const Scalar&       eps,
                const Scalar&       momentum
            )
    {
        const long ncols  = in.cols();
        const long nrows  = in.rows();

        Eigen::VectorXd mu = ( Scalar(1.0) / static_cast<Scalar>(ncols) ) * in.rowwise().sum();

        assert(mu.cols() == m_mean.cols());
        assert(mu.rows() == m_mean.rows());

        // вычисление скользящего среднего мат ожидания
        m_mean = momentum * m_mean + (1 - momentum) * mu;

        inmu = in.colwise() - mu;

        Matrix carre = inmu.array().square();

        Eigen::VectorXd var =  ( Scalar(1.0) / static_cast<Scalar>(ncols) ) * carre.rowwise().sum();

        assert(var.cols() == m_var.cols());
        assert(var.rows() == m_var.rows());

        // вычисление скользящего среднего дисперсии
        m_var = momentum * m_var + (1 - momentum) * var;

        Eigen::VectorXd sqrtvar = (var.array() + eps).array().sqrt();

        invvar = Scalar(1.0) / sqrtvar.array();

        // final
        in_hat = inmu.array().colwise() * invvar.array();
    }

    inline void compute_statistic_graph1D_eval(
            Matrix&             in_hat,
            Eigen::VectorXd&    m_mean,
            Eigen::VectorXd&    m_var,
            const Scalar&       eps
    )
    {
        // compute xmu
        in_hat = in_hat.colwise() - m_mean;

        // compute denominator mean var
        Eigen::VectorXd  denom_var = Scalar(1.0) / (m_var.array() + eps).array().sqrt();

        // final
        in_hat = in_hat.array().colwise() * denom_var.array();
    }

    inline void compute_backward_graph1D(
            const Matrix&       next_layer_data,
            Matrix&             dLz,
            Matrix&             m_z,
            Eigen::VectorXd&    m_gammas,

            Matrix&             m_din,
            Eigen::VectorXd&    m_dg,
            Eigen::VectorXd&    m_db,

            Eigen::VectorXd&    invvar,
            Matrix&             inmu
            )
    {
        Scalar     ncols  = static_cast<Scalar>(m_z.cols());

        m_db = next_layer_data.rowwise().sum();
        m_dg = (m_z.cwiseProduct(next_layer_data)).rowwise().sum();

        Eigen::VectorXd invvar_square                   = invvar.array().square();
        Eigen::VectorXd inmu_nld_prod                   = (inmu.cwiseProduct(next_layer_data)).rowwise().sum();

        Eigen::VectorXd gamma_invvar_prod               = m_gammas.cwiseProduct(invvar);
        Matrix          inmu_invvar_sqr_prod            = inmu.array().colwise() * invvar_square.array();
        Matrix          inmu_invvar_sqr_inmu_nld_prod   = inmu_invvar_sqr_prod.array().colwise() * inmu_nld_prod.array();

        Matrix tmp_nld  = ncols * next_layer_data;

        tmp_nld         = tmp_nld.colwise() - m_db;

        tmp_nld         = tmp_nld.cwiseProduct(inmu_invvar_sqr_inmu_nld_prod);
        tmp_nld         = tmp_nld.cwiseProduct(dLz);

        m_din           = tmp_nld.array().colwise() * gamma_invvar_prod.array();
        m_din.noalias() = (1.0 / ncols) * m_din;

//        m_din = (1.0 / ncols) * m_gammas * invvar *
//                (ncols * next_layer_data - m_db - inmu * invvar_square * inmu_product_nld) * dLz;
    }

}

#endif //XSDNN_INCLUDE_BATCHNORMUTIL_H
