//
// Created by Andrei R. on 16.01.23.
// Copyright (c) 2023 xsDNN. All rights reserved.
//

#ifndef XSDNN_MATH_HPP
#define XSDNN_MATH_HPP

namespace xsdnn {

using DimPair = Eigen::IndexPair<Eigen::DenseIndex>;
using DimArray = std::array<DimPair, 1>;

// must be only static member funtion class
class tensorize {
public:
    template<typename Device, typename In1, typename In2, typename Out>
    static void matmul(Device d, In1& in1, In2& in2, Out& out, const DimArray& dim) {
        out.device(d) = in1.contract(in2, dim);
    }
};

} // xsdnn

// TODO: impl MatMul via static method of some functor?

// TODO: impl MatMul with allocate check






#endif //XSDNN_MATH_HPP
