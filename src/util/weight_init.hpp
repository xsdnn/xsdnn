//
// Created by Andrei R. on 31.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_WEIGHT_INIT_HPP
#define XSDNN_WEIGHT_INIT_HPP


namespace xsdnn {
namespace weight_init {

class function {
public:
    function(Scalar scale) : scale_(scale) {}
    virtual ~function() {}
    virtual void fill(Scalar* data, size_t size, size_t fan_in, size_t fan_out) = 0;

protected:
    Scalar scale_;
};

class xavier : public function {
public:
    xavier() : function(Scalar(6.0)) {}
    explicit xavier(Scalar scale) : function(scale) {}

    virtual void fill(Scalar* data, size_t size, size_t fan_in, size_t fan_out) override {
        const Scalar bias = std::sqrt(scale_ / (fan_in + fan_out));
        uniform_rand(data, size, -bias, bias);
    }
};

class constant : public function {
public:
    constant() : function(Scalar(0.0)) {}
    explicit constant(Scalar scale) : function(scale) {}

    virtual void fill(Scalar* data, size_t size, size_t, size_t) override {
        // TODO: подумать, как можно не обращать внимания на неиспользуемые параметры функции (через макрос)
        tensor_fill(data, size, scale_);
    }
};

} // weight_init
} // xsdnn


#endif //XSDNN_WEIGHT_INIT_HPP
