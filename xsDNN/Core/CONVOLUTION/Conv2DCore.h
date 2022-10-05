//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_CONVCORE_H
#define XSDNN_CONVCORE_H

namespace xsdnn {
    namespace internal {
        namespace conv2d {
            /// Эта функция отвечает за выбор алгоритма прямого прохода свертки. Обязательно передается алгоритм
            /// \n Например core::conv::mec
            /// \param ...
            inline void computeForward(...);
        } // end namespace conv2d
    } // end namespace internal
} // namespace xsdnn


#endif //XSDNN_CONVCORE_H
