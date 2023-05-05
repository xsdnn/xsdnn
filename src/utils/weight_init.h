//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_WEIGHT_INIT_H
#define XSDNN_WEIGHT_INIT_H

#include "../mmpack/inc/mmpack.h"
#include <cmath>
#include "random.h"
#include "tensor_utils.h"

using namespace mmpack;

namespace xsdnn {
    namespace weight_init {

class function {
public:
    function(mm_scalar scale) : scale_(scale) {}
    virtual ~function() {}
    virtual void fill(mm_scalar* data, size_t size, size_t fan_in, size_t fan_out) = 0;

protected:
    mm_scalar scale_;
};

class xavier : public function {
public:
    xavier() : function(mm_scalar(6.0)) {}
    explicit xavier(mm_scalar scale) : function(scale) {}
    virtual void fill(mm_scalar* data, size_t size, size_t fan_in, size_t fan_out) override;
};

class constant : public function {
public:
    constant() : function(mm_scalar(0.0)) {}
    explicit constant(mm_scalar scale) : function(scale) {}
    virtual void fill(mm_scalar* data, size_t size, size_t fan_in, size_t fan_out) override;
};

    } // weight_init
} // xsdnn

#endif //XSDNN_WEIGHT_INIT_H
