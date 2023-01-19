//
// Created by Andrei R. on 16.01.23.
// Copyright (c) 2023 xsDNN. All rights reserved.
//

#ifndef XSDNN_MATH_HPP
#define XSDNN_MATH_HPP

namespace xsdnn {

using DimPair = Eigen::IndexPair<Eigen::DenseIndex>;
using DimArray = std::array<DimPair, 1>;
using ReduceArray1D = std::array<Eigen::DenseIndex, 1>;
using ReduceArray2D = std::array<Eigen::DenseIndex, 2>;

// must be only static member funtion class
class tensorize {
public:

    /*
     * Classical matmul-op
     */
    template<typename Device, typename In1, typename In2, typename Out>
    static void matmul(Device& d, In1& in1, In2& in2, Out& out, const DimArray& dim) {
        out.device(d) = in1.contract(in2, dim);
    }

    /*
     * Matmul + reduce op
     */
    template<typename Device, typename In1, typename In2, typename Out,
             typename DimReduce>
    static void matmul(Device& d, In1& in1, In2& in2, Out& out,
                       const DimArray& dim, const DimReduce& dim_reduce) {
        out.device(d) = in1.contract(in2, dim).maximum(dim_reduce);
    }
};

} // xsdnn

#endif //XSDNN_MATH_HPP
