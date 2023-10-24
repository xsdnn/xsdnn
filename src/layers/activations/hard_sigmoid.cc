//
// Created by rozhin on 14.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/activations/hard_sigmoid.h>

namespace xsdnn {

std::pair<mm_scalar, mm_scalar> hard_sigmoid::out_value_range() const {
    return std::make_pair(mm_scalar (0.1), mm_scalar (0.9));
}

std::string hard_sigmoid::layer_type() const {
    return "hard_sigmoid";
}

void hard_sigmoid::init_params() {
    p_.params.ActivationType = mmpack::HardSigmoid;
    p_.params.Parameters.HardSigmoid.alpha = alpha_;
    p_.params.Parameters.HardSigmoid.beta = beta_;
    ACTIVATION_INIT_PARAMS_LAST_LINE
}

}