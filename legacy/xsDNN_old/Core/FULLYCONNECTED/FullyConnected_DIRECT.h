//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//


#ifndef XSDNN_FULLYCONNECTED_DIRECT_H
#define XSDNN_FULLYCONNECTED_DIRECT_H

namespace xsdnn {
    namespace internal {
        namespace fc {
            namespace algorithm {
                template<typename Activation>
                inline void compute_forward_direct(const xsTypes::Matrix& prev_data,
                                                   xsTypes::Matrix& w,
                                                   xsTypes::Vector& b,
                                                   xsTypes::Matrix& z,
                                                   xsTypes::Matrix& a,
                                                   bool    bias,
                                                   const int out_size) {
                    const long ncols = prev_data.cols();

                    z.resize(out_size, ncols);
                    z.noalias() = w.transpose() * prev_data;
                    if (bias) { z.colwise() += b; }
                    a.resize(out_size, ncols);
                    Activation::activate(z, a);
                }

                template<typename Activation>
                inline void compute_backward_direct(const xsTypes::Matrix& prev_data,
                                                    const xsTypes::Matrix& next_grad,
                                                    xsTypes::Matrix& w,
                                                    xsTypes::Matrix& dw,
                                                    xsTypes::Vector& db,
                                                    xsTypes::Matrix& din,
                                                    xsTypes::Matrix& z,
                                                    xsTypes::Matrix& a,
                                                    bool    bias,
                                                    const int in_size) {
                    const long ncols = prev_data.cols();

                    xsTypes::Matrix &dLz = z;
                    Activation::apply_jacobian(z, a, next_grad, dLz);
                    dw.noalias() = prev_data * dLz.transpose() / ncols;
                    if (bias) { db.noalias() = dLz.rowwise().mean(); }
                    din.resize(in_size, ncols);
                    din.noalias() = w * dLz;
                }
            } // namespace algorithm
        } // namespace fc
    } // namespace internal
} // namespace xsdnn


#endif //XSDNN_FULLYCONNECTED_DIRECT_H
