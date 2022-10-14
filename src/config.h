//
// Created by Andrei R. on 13.10.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_CONFIG_H
#define XSDNN_CONFIG_H

#if defined(DNN_USE_DOUBLE)
typedef double Scalar;
#else
typedef float Scalar;

#endif //XSDNN_CONFIG_H
