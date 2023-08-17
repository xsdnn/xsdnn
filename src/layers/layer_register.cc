//
// Created by rozhin on 25.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <serializer/cerial.h>


/*
 * Save layer register
 */
#define XS_LAYER_SAVE_REGISTER                                  \
XS_LAYER_SAVE_INTERNAL_REGISTER(fully_connected)                \
XS_LAYER_SAVE_INTERNAL_REGISTER(Input)                          \
XS_LAYER_SAVE_INTERNAL_REGISTER(Output)                         \
XS_LAYER_SAVE_INTERNAL_REGISTER(add)                            \
XS_LAYER_SAVE_INTERNAL_REGISTER(abs)                            \
XS_LAYER_SAVE_INTERNAL_REGISTER(acos)                           \
XS_LAYER_SAVE_INTERNAL_REGISTER(and_layer)                      \
XS_LAYER_SAVE_INTERNAL_REGISTER(flatten)                        \
XS_LAYER_SAVE_INTERNAL_REGISTER(relu)                           \
XS_LAYER_SAVE_INTERNAL_REGISTER(max_pooling)                    \
XS_LAYER_SAVE_INTERNAL_REGISTER(global_average_pooling)         \
XS_LAYER_SAVE_INTERNAL_REGISTER(reshape)



/*
 * Load layer register
 */
#define XS_LAYER_LOAD_REGISTER                                  \
cerial c;                                                       \
XS_LAYER_LOAD_INTERNAL_REGISTER(fully_connected)                \
XS_LAYER_LOAD_INTERNAL_REGISTER(Input)                          \
XS_LAYER_LOAD_INTERNAL_REGISTER(Output)                         \
XS_LAYER_LOAD_INTERNAL_REGISTER(add)                            \
XS_LAYER_LOAD_INTERNAL_REGISTER(abs)                            \
XS_LAYER_LOAD_INTERNAL_REGISTER(acos)                           \
XS_LAYER_LOAD_INTERNAL_REGISTER(and_layer)                      \
XS_LAYER_LOAD_INTERNAL_REGISTER(flatten)                        \
XS_LAYER_LOAD_INTERNAL_REGISTER(relu)                           \
XS_LAYER_LOAD_INTERNAL_REGISTER(max_pooling)                    \
XS_LAYER_LOAD_INTERNAL_REGISTER(global_average_pooling)         \
XS_LAYER_LOAD_INTERNAL_REGISTER(reshape)






namespace xsdnn {

void layer_register() {
    XS_LAYER_SAVE_REGISTER
}

void serializer::load(const xs::NodeInfo *node, const xs::TensorInfo *tensor,
                      std::vector<std::shared_ptr<layer>> &owner_nodes) {
    XS_LAYER_LOAD_REGISTER
}

}
