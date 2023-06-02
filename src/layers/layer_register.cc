//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/network.h>
#include <utils/macro.h>

#include <layers/fully_connected.h>

#include <layers/activations/relu.h>

namespace xsdnn {

XS_REGISTER_LAYER_FOR_NET(fully_connected)


/*
 * Activations
 */

XS_REGISTER_LAYER_FOR_NET(relu)

}