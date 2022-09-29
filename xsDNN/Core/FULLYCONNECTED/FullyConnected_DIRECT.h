//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_FULLYCONNECTED_DIRECT_H
#define XSDNN_FULLYCONNECTED_DIRECT_H


namespace internal {
    namespace fc {
        namespace algorithm {
            typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

            template<typename Activation>
            inline void compute_forward_direct(const Matrix& prev_data,
                                               Matrix& w,
                                               Vector& b,
                                               Matrix& z,
                                               Matrix& a,
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
            inline void compute_backward_direct(const Matrix& prev_data,
                                                const Matrix& next_grad,
                                                Matrix& w,
                                                Matrix& dw,
                                                Vector& db,
                                                Matrix& din,
                                                Matrix& z,
                                                Matrix& a,
                                                bool    bias,
                                                const int in_size) {
                const long ncols = prev_data.cols();

                Matrix &dLz = z;
                Activation::apply_jacobian(z, a, next_grad, dLz);
                dw.noalias() = prev_data * dLz.transpose() / ncols;
                if (bias) { db.noalias() = dLz.rowwise().mean(); }
                din.resize(in_size, ncols);
                din.noalias() = w * dLz;
            }
        } // end namespace algorithm
    } // end namespace fc
} // end namespace internal

#endif //XSDNN_FULLYCONNECTED_DIRECT_H
