//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/relu.h>

namespace xsdnn {

std::pair<mm_scalar, mm_scalar> relu::out_value_range() const {
    return {(mm_scalar) 0.1f, (mm_scalar) 0.9f};
}

std::string relu::layer_type() const {
    return "relu";
}

void relu::init_params() {
    p_.params.ActivationType = mmpack::Relu;
    ACTIVATION_INIT_PARAMS_LAST_LINE
}

} // xsdnn