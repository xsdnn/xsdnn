//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_OPTIMIZER_H
#define XSDNN_OPTIMIZER_H

namespace xsdnn {
    /*!
    \brief Родительский класс оптимизаторов
    \author __[shuffle-true](https://github.com/shuffle-true)__
    \version 0.0
    \date Март 2022 года
    */
    class Optimizer {
    public:
        virtual ~Optimizer() = default;


        ///
        /// Сброс информации о текущем оптимайзере
        ///

        virtual void reset() {};

        ///
        /// Собственно метод, отвечающий за обновление весов в сетке
        ///

        virtual void update(xsTypes::AlignedMapVec &dvec, xsTypes::AlignedMapVec &vec) = 0;
    };
} // namespace xsdnn



#endif //XSDNN_OPTIMIZER_H
