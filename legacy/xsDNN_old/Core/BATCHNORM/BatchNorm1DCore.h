//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//


#ifndef XSDNN_BATCHNORM1DCORE_H
#define XSDNN_BATCHNORM1DCORE_H

#include "BatchNorm1D_DIRECT.h"

namespace xsdnn {
    namespace internal {
        namespace bn1d {
            template<typename Activation>
            inline void computeForward(const xsTypes::Matrix& prev_data,
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
            inline void computeBackward(const xsTypes::Matrix& prev_data,
                                        const xsTypes::Matrix& next_grad,
                                        xsTypes::Matrix& z,
                                        xsTypes::Matrix& a,
                                        xsTypes::Matrix& din,
                                        xsTypes::Vector& dg,
                                        xsTypes::Vector& db,
                                        xsTypes::Vector& stddev,
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
        } // namespace bn1d
    } // namespace internal
} // namespace xsdnn


#endif //XSDNN_BATCHNORM1DCORE_H
