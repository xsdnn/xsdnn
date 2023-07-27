//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_MACRO_H
#define XSDNN_MACRO_H
#include <iostream>

#define XS_UNUSED_PARAMETER(x) (void)(x)
#define XS_2D_1D_CONVERTER(i, j, lda) i * lda + j
#define XS_NUM_THREAD 12

#define XS_LAYER_SAVE_INTERNAL_REGISTER(layer_typename)                                             \
serializer::get_instance().register_saver(#layer_typename, save<layer_typename>);                   \

#define XS_LAYER_LOAD_INTERNAL_REGISTER(layer_typename)                                             \
if (node->name() == #layer_typename) {                                                              \
    std::shared_ptr<layer_typename> layer = c.deserialize<layer_typename>(node, tensor);            \
    owner_nodes.push_back(layer);                                                                   \
}

#endif //XSDNN_MACRO_H
