//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_BATCHNORM1D_DIRECT_H
#define XSDNN_BATCHNORM1D_DIRECT_H

# include "../../Utils/Math.h"

namespace xsdnn {
    namespace internal {
        namespace bn1d {
            namespace algorithm {
                template<typename Activation>
                inline void compute_forward_direct(const xsTypes::Matrix& prev_data,
                                                   xsTypes::Matrix& z,
                                                   xsTypes::Matrix& a,
                                                   xsTypes::Vector& m_curr,
                                                   xsTypes::Vector& v_curr,
                                                   xsTypes::Vector& m,
                                                   xsTypes::Vector& v,
                                                   xsTypes::Vector& stddev,
                                                   xsTypes::Vector& g,
                                                   xsTypes::Vector& b,
                                                   std::string& wflow,
                                                   Scalar& eps,
                                                   bool affine,
                                                   const int size) {
                    const long ncols = prev_data.cols();
                    z.resize(size, ncols);
                    a.resize(size, ncols);

                    xsTypes::Vector mean = (wflow == "train") ? m_curr : m;
                    xsTypes::Vector var = (wflow == "train") ? v_curr : v;

                    if (wflow == "train") {
                        internal::math::update_statistics(prev_data, size, mean, var);
                    }

                    stddev = (var.array() + eps).sqrt();

                    int index = 0;
                    for (auto col: prev_data.colwise()) {
                        z.col(index++) = (col.array() - mean.array()) / stddev.array();
                    }

                    if (affine) {
                        z = z.array().colwise() * g.array();
                        z = z.array().colwise() + b.array();
                    }

                    Activation::activate(z, a);
                }

                template<typename Activation>
                inline void compute_backward_direct(const xsTypes::Matrix& prev_data,
                                                    const xsTypes::Matrix& next_grad,
                                                    xsTypes::Matrix& z,
                                                    xsTypes::Matrix& a,
                                                    xsTypes::Matrix& din,
                                                    xsTypes::Vector& dg,
                                                    xsTypes::Vector& db,
                                                    xsTypes::Vector& stddev,
                                                    bool affine,
                                                    const int size) {
                    const int ncols = z.cols();
                    din.resize(size, ncols);

                    xsTypes::Matrix dLz;
                    Activation::apply_jacobian(z, a, next_grad, dLz);

                    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
                    //
                    // dE(L)/dX =
                    //   (dL/dz - mean(dL/dz) - mean(dL/dz \cdot Y) \cdot Y)
                    //     ./ sqrt(var(X) + eps)
                    //

                    xsTypes::Matrix dLz_dot_y;
                    dLz_dot_y.resize(size, ncols);
                    dLz_dot_y = dLz.cwiseProduct(z);

                    xsTypes::Vector mean_dLz, mean_dLz_dot_y;
                    mean_dLz.resize(size);
                    mean_dLz_dot_y.resize(size);
                    mean_dLz = dLz.rowwise().mean();
                    mean_dLz_dot_y = dLz_dot_y.rowwise().mean();

                    int index = 0;
                    for (auto col: din.colwise()) {
                        col = dLz.col(index) - mean_dLz - mean_dLz_dot_y.cwiseProduct(z.col(index));
                        index += 1;
                    }
                    index = 0;
                    for (auto row: din.rowwise()) {
                        row.array() /= stddev[index++];
                    }

                    if (affine) {
                        dg = mean_dLz_dot_y;
                        db = mean_dLz;
                    }
                }

            } // namespace algorithm
        } // namespace bn1d
    } // namespace internal
} // namespace xsdnn


#endif //XSDNN_BATCHNORM1D_DIRECT_H
