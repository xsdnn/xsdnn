//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_OPTIMIZER_H
#define XSDNN_OPTIMIZER_H

/*!
\brief Родительский класс оптимизаторов
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Март 2022 года
*/
class Optimizer {
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Vector::AlignedMapType AlignedMapVec;

public:
    virtual ~Optimizer() = default;


    ///
    /// Сброс информации о текущем оптимайзере
    ///

    virtual void reset() {};

    ///
    /// Собственно метод, отвечающий за обновление весов в сетке
    ///

    virtual void update(AlignedMapVec &dvec, AlignedMapVec &vec) = 0;
};

#endif //XSDNN_OPTIMIZER_H
