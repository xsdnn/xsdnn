//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_BATCHNORM1DCORE_H
#define XSDNN_BATCHNORM1DCORE_H

#include "BatchNorm1D_DIRECT.h"

namespace internal {
    namespace bn1d {
        typedef Eigen::VectorXd Vector;

        template<typename Activation>
        inline void computeForward(const Matrix& prev_data,
                                   Matrix& z,
                                   Matrix& a,
                                   Vector& m_curr,
                                   Vector& v_curr,
                                   Vector& m,
                                   Vector& v,
                                   Vector& stddev,
                                   Vector& g,
                                   Vector& b,
                                   std::string& wflow,
                                   Scalar& eps,
                                   bool affine,
                                   const int size) {
            internal::bn1d::algorithm::compute_forward_direct<Activation>(
                    prev_data,
                    z,a,
                    m_curr,v_curr,
                    m,v,
                    stddev, g, b,
                    wflow, eps,
                    affine, size
                    );
        }

        template<typename Activation>
        inline void computeBackward(const Matrix& prev_data,
                                    const Matrix& next_grad,
                                    Matrix& z,
                                    Matrix& a,
                                    Matrix& din,
                                    Vector& dg,
                                    Vector& db,
                                    Vector& stddev,
                                    bool affine,
                                    const int size) {

            internal::bn1d::algorithm::compute_backward_direct<Activation>(
                    prev_data,next_grad,
                    z,a,
                    din,dg,db,
                    stddev,
                    affine, size
                    );

        }
    } // end namespace bn1d
} // end namespace internal

#endif //XSDNN_BATCHNORM1DCORE_H
